import cv2 as cv
import math
import argparse
from imutils.video import VideoStream
from imutils.video import FPS
#from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

############ Add argument parser for command line arguments ############
parser = argparse.ArgumentParser(description='Use this Text Detector (https://arxiv.org/abs/1704.03155v2)')
parser.add_argument("-input","--input", 
					help='Path to input video file. Skip this argument to capture frames from a camera.')
parser.add_argument('--model', required=True, default='C:/Users/HP/Desktop/opencv-text-detection/frozen_east_text_detection.pb',
                    help='Path to a binary .pb file of model contains trained weights.')
parser.add_argument('--width', type=int, default=320,
                    help='Preprocess input image by resizing to a specific width. It should be multiple by 32.')
parser.add_argument('--height',type=int, default=320,
                    help='Preprocess input image by resizing to a specific height. It should be multiple by 32.')
parser.add_argument('--thr',type=float, default=0.5,
                    help='Confidence threshold.')
parser.add_argument('--nms',type=float, default=0.4,
                    help='Non-maximum suppression threshold.')
args = parser.parse_args()

def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"

    height = scores.shape[2]
    width = scores.shape[3]

    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]

def main():
    # Read and store arguments
    confThreshold = args.thr
    nmsThreshold = args.nms
    inpWidth = args.width
    inpHeight = args.height
    model = args.model

    # Load network
    net = cv.dnn.readNet(model)

    # Create a new named window
    kWinName = "Scene Text Detector"
    cv.namedWindow(kWinName, cv.WINDOW_NORMAL)
    layerNames = []
    #output sigmoid activation which gives us the probability of a region containing text or not
    layerNames.append("feature_fusion/Conv_7/Sigmoid")
    #output feature map that represents the “geometry” of the image 
    layerNames.append("feature_fusion/concat_3")

    print ("--- type q to leave text recognition ---")
    # start the FPS throughput estimator
    fps = FPS().start()
    
    if not args.input :
        print("--- starting video stream ---")
    else :
        print("--- oppenig selected video ---")
        
    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)
    
    while True :
        # Read frame
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        # Get frame height and width
        height_ = frame.shape[0]
        width_ = frame.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)

        # Create a 4D blob from frame.
        blob = cv.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        outs = net.forward(layerNames)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = decode(scores, geometry, confThreshold)

        # Apply NMS
        indices = cv.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv.boxPoints(boxes[i[0]])
            
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (int(vertices[j][0]), int(vertices[j][1]))
                p2 = (int(vertices[(j + 1) % 4][0]), int(vertices[(j + 1) % 4][1]))
                cv.line(frame, p1, p2, (0, 255, 0), 1)

            rect = cv.minAreaRect(vertices)
            box = cv.boxPoints(rect)
            box=vertices
            box = np.int0(box)
            #cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

             # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height-1],[0, 0], [width-1, 0],[width-1, height-1]], dtype="float32")

             # the perspective transformation matrix
            M = cv.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            if width > height : 
            	warped = cv.warpPerspective(frame, M, (width, height))
            else :
            	warped = cv.warpPerspective(frame, M, (height, width))

            cv.imwrite("test_crop_img.jpg", warped)
            img = cv.imread("test_crop_img.jpg")
            #cv.imshow("detect",img)

            # in order to apply Tesseract v4 to OCR text we must supply
            # (1) a language, (2) an OEM flag of 4, indicating that the we
            # wish to use the LSTM neural net model for OCR, and finally
            # (3) an OEM value, in this case, 7 which implies that we are
            # treating the ROI as a single line of text
            config = ("-l eng --oem 1 --psm 7")
            text = pytesseract.image_to_string(img,config =config)
            if text :
                print(text)
            cv.putText(frame, text, (int(vertices[1][0]), int(vertices[1][1]) - 20),cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Put efficiency information
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        # Display the frame
        cv.imshow(kWinName,frame)
        fps.update()
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        fps.stop()
        # if we are using a webcam, release the pointer

if __name__ == "__main__":
    main()