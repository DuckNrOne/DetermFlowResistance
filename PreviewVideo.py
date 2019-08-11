import cv2 as cv
import numpy as np


def show_preview_frame(img):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([150, 120, 140])
    mask1 = cv.inRange(img, lower_black, upper_black)

    return mask1


if __name__ == "__main__":  
    inputVideo = input('Input File: ')

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inputVideo))
    if not capture.isOpened:
        print('Unable to open: ' + inputVideo)

    count = 0

    while count < 125:
        _, frame = capture.read()
        if frame is None:
            break

        cv.imshow('Preview', show_preview_frame(frame))
        count += 1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break



