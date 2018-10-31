# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import configparser
from utils import *

config = configparser.ConfigParser()
config.read("./src/config.ini")

LOWER_BOUND = config.getint("DEFAULT", "LOWER_BOUND")
UPPER_BOUND = config.getint("DEFAULT", "UPPER_BOUND")
FILTERED = config.getboolean("DEFAULT", "FILTERED")
VIDEO = config.get("DEFAULT", "VIDEO")

cap = cv2.VideoCapture(VIDEO)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

data = [np.zeros((1,length)),np.zeros((1,length)),np.zeros((1,length))]
transformed = [None, None, None]

r = np.zeros((1,length))
g = np.zeros((1,length))
b = np.zeros((1,length))


k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        data[0][0,k] = np.mean(frame[330:360,610:640,0])
        data[1][0,k] = np.mean(frame[330:360,610:640,1])
        data[2][0,k] = np.mean(frame[330:360,610:640,2])
    else:
        break
    k = k + 1


cap.release()
cv2.destroyAllWindows()

n = 1024
if n > length:
    n = 2**int(np.log2(length))
f = np.linspace(-n/2,n/2-1,n)*fps/n


for k in range(0,3):
    data[k] = data[k][0,0:n] - np.mean(data[k][0,0:n])
    transformed[k] = np.abs(fftshift(fft(data[k])))**2
    if FILTERED:
        bandpass_filter(LOWER_BOUND, UPPER_BOUND, transformed[k], f * 60)

    plt.plot(60*f, transformed[k])
    plt.xlabel("frecuencia [1/minuto]")
    plt.xlim(0,200)
    plt.show()


print("Frecuencia cardiaca en R: ", abs(f[np.argmax(transformed[0])])*60, " pulsaciones por minuto")
print("Frecuencia cardiaca en G: ", abs(f[np.argmax(transformed[1])])*60, " pulsaciones por minuto")
print("Frecuencia cardiaca en B: ", abs(f[np.argmax(transformed[2])])*60, " pulsaciones por minuto")