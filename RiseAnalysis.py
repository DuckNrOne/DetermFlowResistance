import cv2 as cv
import math
from statistics import mean
from statistics import stdev
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats


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


refPt = []


def preview_video():
    """
    Preview of analysing video
    """
    inputVideo = input('Input File: ')

    capture = cv.VideoCapture(cv.samples.findFileOrKeep(inputVideo))
    if not capture.isOpened:
        print('Unable to open: ' + inputVideo)
    
    _, frame_temp = capture.read()
    height, width, _ = frame_temp.shape    
    height = 960/height
    width = 1920/width

    while True:
        _, frame = capture.read()
        if frame is None:
            break

        frame = cv.resize(frame, None, fx=height, fy=height)
        cv.imshow('Preview', show_preview_frame(frame))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


lower_black = np.array([0, 0, 0])
upper_black = np.array([90, 110, 110])


def show_preview_frame(img):
    """
    Preview of a single frame
    """
    global lower_black, upper_black
    mask1 = cv.inRange(img, lower_black, upper_black)

    return mask1



def start_analysis_folder():
    """
    Calculation for a folder with videos
    """
    inputFolder = input('Folder: ')
    files = os.listdir(inputFolder)
    for file in files:
        if file[len(file) - 3:] == 'mp4': analyse_video(inputFolder + "\\" + file)


def start_analysis_file():
    """
    Calculation of a single video
    """
    inputVideo = input('File: ')
    analyse_video(inputVideo)


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

    cv.imshow('Reference Point', frame_temp)
    cv.setMouseCallback('Reference Point', click_and_crop)
    cv.waitKey()
    cv.destroyWindow('Reference Point')

    # Resize reference points
    """
    global refPt
    refPt[0][0] = int(refPt[0][0] / width)
    refPt[1][0] = int(refPt[1][0] / width)
    refPt[0][1] = int(refPt[0][1] / height)
    refPt[1][1] = int(refPt[1][1] / height)
    print(refPt)
    """
    frame_values = {}
    frames = 0
    fps = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    percent = 0
    print(fps)

    while True:
        _, frame = capture.read()
        if frame is None:
            break

        frame = cv.resize(frame, None, fx=height, fy=width)
        if frames % 4 == 0:
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

    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #
    # lower_black = np.array([180, 120, 70])
    # upper_black = np.array([10, 255, 255])
    # mask1 = cv.inRange(hsv, lower_black, upper_black)

    # lower_red = np.array([0, 120, 70])
    # upper_red = np.array([10, 255, 255])
    # mask1 = cv.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170, 120, 70])
    # upper_red = np.array([180, 255, 255])
    # mask2 = cv.inRange(hsv, lower_red, upper_red)
    #
    # mask1 = mask1 + mask2
    global refPt
    points = []

    for x in range(refPt[0][0], refPt[1][0]):
        for y in range(refPt[0][1], refPt[1][1]):
            if ([0, 0, 0] != mask1[y, x]).all():
                points.append([x, y])

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
    """
    Creating the data
    """
    data = pd.DataFrame(columns=['Frame', 'Angle'])
    i = 0
    for val in dict_degree.keys():
        data.loc[i] = [val] + [dict_degree[val]]
        i += 1

    # print(data)
    data.plot.scatter(x='Frame', y='Angle')
    plt.show()
    data['Angle'].plot.hist(grid=True, bins=20, rwidth=0.9)
    plt.show()


def flowspeed_to_speed(flow_speed: float):
    """
    Converting from flowspeed to normal speed
    """
    return (4*flow_speed)/(60000*math.pi*0.044**2)
