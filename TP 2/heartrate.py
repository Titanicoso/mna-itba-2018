# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import signal
from utils import *

cap = cv2.VideoCapture('2017-09-14 21.53.59.mp4')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

hasFilter = True

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

low = 40 / (0.5 * fps * 60)
high = 120 / (0.5 * fps * 60)
s, d= signal.bessel(6,[low, high], btype='band')

for k in range(0,3):
    data[k] = data[k][0,0:n] - np.mean(data[k][0,0:n])
    if hasFilter:
        data[k] = signal.lfilter(s,d, data[k])


    transformed[k] = np.abs(np.fft.fftshift(fft(data[k])))**2

    plt.plot(60*f, transformed[k])
    plt.xlabel("frecuencia [1/minuto]")
    plt.xlim(0,200)
    plt.show()


print("Frecuencia cardíaca en R: ", abs(f[np.argmax(transformed[0])])*60, " pulsaciones por minuto")
print("Frecuencia cardíaca en G: ", abs(f[np.argmax(transformed[1])])*60, " pulsaciones por minuto")
print("Frecuencia cardíaca en B: ", abs(f[np.argmax(transformed[2])])*60, " pulsaciones por minuto")