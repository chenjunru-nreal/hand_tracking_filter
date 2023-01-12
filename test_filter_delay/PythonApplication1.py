import numpy as np
import matplotlib.pyplot as plt
from math import *
import random

from OneEuroFilter import OneEuroFilter


class OneEuroFilterProto:
    __Beta = 0
    __MinCutoff = 0
    __DCutOff = 1.0
    __prev = [0.0, np.array([0, 1, 0]), np.zeros(3)]

    def __init__(self, mincutoff, beta, dcutoff):
        self.__Beta = beta
        self.__MinCutoff = mincutoff
        self.__DCutOff = dcutoff

    @staticmethod
    def Alpha(dt, cutoff):
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1)

    def Step(self, t, x):
        dt = t - self.__prev[0]
        dx = (x - self.__prev[1]) / dt
        a = OneEuroFilterProto.Alpha(dt, self.__DCutOff)
        dx_res = self.__prev[2] * (1 - a) + dx * a
        cutoff = self.__MinCutoff + self.__Beta * np.sqrt(sum(dx_res ** 2))
        a = OneEuroFilterProto.Alpha(dt, cutoff)
        x_res = self.__prev[1] * (1 - a) + x * a
        self.__prev = [t, x_res, dx_res]
        return x_res


def CalculateFilterDelay(mincutoff, beta, dcutoff, speed):
    # f = OneEuroFilter(
    #     np.array([0, 1, 0], dtype=np.float64), 0.0, mincutoff, beta, dcutoff
    # )
    f = OneEuroFilterProto(mincutoff, beta, dcutoff)
    direction = np.array([0, 1, 0], dtype=np.float64)
    
    direction_list_x = []
    direction_list_y = []
    dt = 0.0167
    angle = speed * dt
    for time in range(1, 1001):
        x = direction[0] * cos(radians(angle)) - direction[1] * sin(radians(angle))
        y = direction[0] * sin(radians(angle)) + direction[1] * cos(radians(angle))
        direction[0] = x
        direction[1] = y
        directionFilter = f.Step(time * dt, direction)
        direction_list_x.append(x)
        direction_list_y.append(y)
    # if speed == 359:
    #     fig = plt.figure()  # 生成窗口
    #     plt.plot(direction_list_x, direction_list_y)

    # return Angle(directionFilter, direction) / speed * 1000
    return Angle(directionFilter, direction) / speed * 1000


def Angle(v1, v2):
    dot = np.dot(v1, v2)
    arccos = np.arccos(dot / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(arccos)


x = range(1, 360)
y = []
for i in x:
    y.append(CalculateFilterDelay(0.001, 10.0, 1.0, i))
    
x_label = [f / 60 for f in x]

fig, ax = plt.subplots()
plt.title("speed - time delay")
ax.plot(x_label, y)
plt.xlabel("speed (deg/s)")
plt.ylabel("Time (ms)")

plt.show()
