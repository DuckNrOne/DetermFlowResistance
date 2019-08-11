import cv2 as cv
import RiseAnalysis as ra
import numpy as np
import math

r = 0
diam = 480.0


def edit_Frame(img):
    
    global r, diam
    imgedit = np.zeros((1080,1920,3))

    points = []
    for i in range(0, 1079):
        for j in range(0, 1919):
            if ([0, 0, 0] != img[i, j]).all():
                points.append([j, get_real_height(i)])
            
    points = list(filter(lambda v: v[0] < 1080 & v[0] >= 0, points))
    for point in points:
        imgedit[point[1], point[0]] = [255, 255, 255]
    
    return imgedit    


def get_real_height(hTemp: float):
    global diam, r
    h = diam - hTemp
    try:
        return h/(1.33 * math.cos(math.asin((h/(r*1.33))) - math.asin(h/r)))
    except:
        return -1


if __name__ == "__main__":
    inputVideo = input('Video: ')
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inputVideo))
    if not capture.isOpened:
        print('Unable to open: ' + inputVideo)

    # First step to resize preview image
    _, frame_temp = capture.read()
    height, width, _ = frame_temp.shape
    height = 1080/height
    width = 1920/width
    frame_temp = cv.resize(frame_temp, None, fx=height, fy=width)

    # Get Diameter height
    cv.imshow('Diameter Height', frame_temp)
    cv.setMouseCallback('Diameter Height', ra.diam_height)
    cv.waitKey()
    cv.destroyWindow('Diameter Height')

    while True:
        _, frame = capture.read()
        if frame is None:
            break

        frame = cv.resize(frame, None, fx=height, fy=width)
        frame = ra.preview_frame(frame)
        frame_edit = edit_Frame(frame)

        print(frame_edit.shape)

        cv.imshow('Masked Glass', frame)
        cv.imshow('Refraction Glass', frame_edit)
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

