# -*- coding: utf-8 -*-
import numpy as np

#https://en.wikipedia.org/wiki/Cooley–Tukey_FFT_algorithm code adapted to remove for
def fft(x):
    n = x.shape[0]
    if n == 1:
        return x
    even = fft(x[::2])
    odd = fft(x[1::2])

    exponential = np.exp(-2j * np.pi * np.arange(n/2) / n)
    ret = even + exponential * odd
    ret = np.concatenate([ret, even - exponential * odd])
    return ret