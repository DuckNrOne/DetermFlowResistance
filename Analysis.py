"""
Script to analyse and evaluate a measured series
"""
import RiseAnalysis as ra
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import math
from statistics import mean
from statistics import stdev
import pandas as pd

print('[[START]]')
folder = input('Folder: ')
weight = float(input('Weight [kg]: '))

arr_deg = np.array([])
arr_spd = np.array([])
arr_stdev = np.array([])

for filename in os.listdir(folder):
    if filename.endswith(".MP4"):
        print(filename)
        speed = ra.flowspeed_to_speed(float(input('Flow speed [l/min]: ')))

        values = ra.analyse_video(folder + "\\" + filename)

        np.append(arr_stdev, [stdev(values)])
        np.append(arr_deg, [mean(values)])
        np.append(arr_spd, [speed])
    else:
        continue

pd.DataFrame({"Speed [m/s": arr_spd, "Winkel [°]": arr_deg}).to_excel("Raw_Data.xlsx")

arr_tan = 1 / np.tan(np.radians(arr_deg))
arr_spdq = 1 / np.square(arr_spd)

plt.plot(arr_spdq, arr_tan)
plt.xlabel("1/v²")
plt.ylabel("1/tan(a)")
plt.show()

arr_tan = arr_tan.reshape(-1, 1)
model = LinearRegression()
model.fit(arr_spdq, arr_tan)

a = model.intercept_
b = model.coef_

arr_fa = math.sqrt(-b*9.81*weight)*np.square(arr_spd)
arr_fw = (9.81 * weight * np.square(arr_spd)) / a

arr_spd = np.append(arr_spd, [0])
arr_spd = np.append(arr_deg, [0])
arr_spd = np.append(arr_stdev, [90])
arr_spd = np.append(arr_fw, [0])
arr_spd = np.append(arr_fa, [1000*9.81*(4/3)*math.pi*0.00085**3])

data = pd.DataFrame({"Fließgeschwindigkeit [m/s]": arr_spd , 'Winkel [°]': arr_deg, 'Standardabweichung [°]': arr_stdev, 'Gewichtskraft [N]': [np.full(1,len(arr_stdev), 9.81*weight)], 'Auftriebkraft [N]': arr_fa, 'Strömungswiderstand [N]': arr_fw})
data.to_excel("OutPut.xlsx")

data.plot(x='Fließgeschwindigkeit [m/s]', y='Auftriebkraft [N]')
plt.xlabel("Fließgeschwindigkeit [m/s")
plt.ylabel("Auftriebkraft [N]")
plt.show()

data.plot(x='Fließgeschwindigkeit [m/s]', y='Strömungswiderstand [N]')
plt.xlabel("Fließgeschwindigkeit [m/s")
plt.ylabel("Strömungswiderstand [N]")
plt.show()
