import cv2 as cv
import RiseAnalysis as ra

inputVideo = input('Input File: ')

capture = cv.VideoCapture(cv.samples.findFileOrKeep(inputVideo))
if not capture.isOpened:
    print('Unable to open: ' + inputVideo)

_, frame = capture.read()
frame = cv.resize(frame, None, fx=0.5, fy=0.5)
cv.imshow('Preview1', frame)
cv.waitKey()
cv.imshow('Preview2', ra.show_preview_frame(frame))
cv.waitKey()


    