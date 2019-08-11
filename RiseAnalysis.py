import cv2 as cv
import math
from statistics import mean
from statistics import stdev
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats


refPt = []
r = 0
diam = 480.0
lower_black = np.array([0, 0, 0])
upper_black = np.array([100, 100, 110])


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv.EVENT_LBUTTONDOWN:
        refPt = [[x, y]]

    # check to see if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append([x, y])


def diam_height(event, x, y, flags, param):
    # grab references to the global variables
    global diam, r

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv.EVENT_LBUTTONDOWN:
        diam = y

    # check to see if the left mouse button was released
    elif event == cv.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        r = abs(diam - y)
        diam = (diam + y)/2


def get_boundaries(inputVideo: str):

    global upper_black

    print('DEF BOUNDARIES')
    print('Press q to quit.')
    inputArray = input("Upper boundary: ")

    while inputArray != 'q':
        upper_black = np.array(list(map(int, inputArray.split())))
        preview_video(inputVideo)
        inputArray = input("Upper boundary: ")


def preview_video(inputVideo: str):
    """
    Preview of analysing video
    """

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inputVideo))
    if not capture.isOpened:
        print('Unable to open: ' + inputVideo)
    
    _, frame_temp = capture.read()
    height, width, _ = frame_temp.shape    
    height = 1080/height
    width = 1920/width

    _, frame = capture.read()

    frame = cv.resize(frame, None, fx=height, fy=height)
    cv.imshow('Preview', preview_frame(frame))
    cv.waitKey()

    cv.destroyWindow('Preview')


def preview_frame(img):
    """
    Preview of a single frame
    """
    global lower_black, upper_black
    mask1 = cv.inRange(img, lower_black, upper_black)

    return mask1


def analyse_video(inputVideo: str):
    """
    Iteration through a video
    """
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inputVideo))
    if not capture.isOpened:
        print('Unable to open: ' + inputVideo)

    # First step to resize preview image
    _, frame_temp = capture.read()
    height, width, _ = frame_temp.shape
    height = 960/height
    width = 1920/width
    frame_temp = cv.resize(frame_temp, None, fx=height, fy=width)

    # Get Reference Point
    cv.imshow('Reference Point', frame_temp)
    cv.setMouseCallback('Reference Point', click_and_crop)
    cv.waitKey()
    cv.destroyWindow('Reference Point')

    # Get Diameter height
    cv.imshow('Diameter Height', frame_temp)
    cv.setMouseCallback('Diameter Height', diam_height)
    cv.waitKey()
    cv.destroyWindow('Diameter Height')


    frame_values = {}
    frames = 0
    fps = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    percent = 0
    global refPt, diam, r
    print("Reference points:", refPt)
    print("Diameter height:", diam)
    print("Radius: ", r)
    print("Pictures:", fps)

    while True:
        _, frame = capture.read()
        if frame is None:
            break

        frame = cv.resize(frame, None, fx=height, fy=width)
        if frames % 5 == 0:
            frame_values[frames] = analysis_frame(frame)

        frames += 1
        if (frames/fps) * 100 > percent:
            print(str(percent) + "%")
            percent += 10
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    frame_values = {k: v for k, v in frame_values.items() if v != 8000}

    print('Bevor Sdtev')
    create_data(frame_values, str(inputVideo + '_BevorStDev'))

    # Standard-Abweichung I >>> Winkel
    while True:
        length = len(frame_values)
        frame_values = {k: v for k, v in frame_values.items() if
                        mean(frame_values.values()) + 3 * stdev(frame_values.values()) >= v>= mean(frame_values.values()) - 3 * stdev(
                            frame_values.values())}
        if length == len(frame_values):
            break

    print("ERGEBNISSE\n"
          "Winkel Durchschnitt: " + str(mean(frame_values.values())) + "°\n"
            "Winkel min: " + str(min(frame_values.values())) + "°\n"
            "Winkel max: " + str(max(frame_values.values())) + "°\n"
          "StDev: " + str(stdev(frame_values.values())) + "°\n")

    print('After Stdev')
    create_data(frame_values, str(inputVideo + 'AfterStDev'))
    return np.array(frame_values.values())


def analysis_frame(img):
    """
    Analyse of a single frame
    """
    global lower_black
    global upper_black
    mask1 = cv.inRange(img, lower_black, upper_black)

    global refPt
    points = []

    for x in range(refPt[0][0], refPt[1][0]):
        for y in range(refPt[0][1], refPt[1][1]):
            if ([0, 0, 0] != mask1[y, x]).all():
                points.append([x, get_real_height(y)])

    if len(points) < 2:
        return 8000

    slope, intercept, _, _, _ = stats.linregress(list(map(lambda x: x[0], points)), list(map(lambda y: y[1], points)))
    pre = slope

    f = lambda fq: fq * slope + intercept
    fx = lambda x, y: (y*slope + x - slope * intercept)/(slope**2+1)
    fd = lambda x, y: math.sqrt((fx(x, y) - x)**2 + (f(fx(x, y) - y))**2)

    count = 0

    while count < 15:
        dif_list = list(map(lambda arr: fd(arr[0], arr[1]), points))
        points = list(filter(lambda v: mean(dif_list) + 2 * stdev(dif_list) >= fd(v[0], v[1]) >= mean(dif_list) - 2 * stdev(dif_list), points))
        if len(points) < 2:
            return 8000
        slope, intercept, _, _, _ = stats.linregress(list(map(lambda x: x[0], points)), list(map(lambda y: y[1], points)))
        if round(slope, 4) == round(pre, 4):
            break
        else:
            pre = slope
            count += 1

    return math.degrees(math.atan(slope))


def create_data(dict_degree: dict, name: str):

    plt.plot(list(dict_degree.keys()), list(dict_degree.values()), 'ro')
    plt.xlabel('Frame')
    plt.ylabel('Angle [deg]')
    plt.show()
    plt.hist(dict_degree.values(), normed=True, bins=15)
    plt.xlabel('Angle [deg]')
    plt.show()


def flowspeed_to_speed(flow_speed: float):
    """
    Converting from flowspeed to normal speed
    """
    return (4*flow_speed)/(60000*math.pi*0.044**2)


def get_real_height(hTemp: float):
    global diam, r
    h = diam - hTemp
    return h/(1.33 * math.cos(math.asin((h/(r*1.33))) - math.asin(h/r)))


if __name__ == "__main__":
    print("PROCESS STARETD")
    inputVideo = input('Video:')
    get_boundaries(inputVideo)
    analyse_video(inputVideo)

