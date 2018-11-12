# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 00:00:46 2018

@author: Shauvik
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.fftpack import dct,idct
from scipy.fftpack import fft,ifft

def ceil(x):
    integer_part = np.int(x)
    float_part = x - integer_part
    ceil_part = x + (1-float_part)
    return int(ceil_part)

def floor(x):
    return np.int(x)

def meanSquaredError(s1,s2):
    arr = abs((s1-s2))
    sum = 0.0
    for i in range(len(arr)):
        sum = sum + (arr[i]*arr[i])
    sum = sum/np.float(len(arr))
    return sum


[rate,data] = scipy.io.wavfile.read("song.wav", mmap=False)


t_start = 0
t_end = (t_start + 1)

N_ = 44100
start_index = t_start * N_
end_index = t_end * N_
signal = data[start_index:end_index]
l_channel = signal[:,0]
r_channel = signal[:,1]


x_l = l_channel
x_r = r_channel


X = np.loadtxt('compressedFile.txt')
N = np.int(X[0,0])
M = np.int(X[0,1])
print('N = '+str(N)+'     M = '+str(M))

XDCT_L_m = np.pad(X[1:,0], (0, M), 'constant')
XDCT_R_m = np.pad(X[1:,1], (0, M), 'constant')
xdct_l_m = scipy.fftpack.idct(XDCT_L_m,norm = 'ortho')
xdct_r_m = scipy.fftpack.idct(XDCT_R_m,norm = 'ortho')
signal_decompressed = np.vstack((xdct_l_m, xdct_r_m)).T

mse_l = meanSquaredError(x_l,xdct_l_m)
mse_r = meanSquaredError(x_r,xdct_r_m)
print('MSE Left Channel = '+str(mse_l))
print('MSE Right Channel = '+str(mse_r))
idx_l = np.argmax(abs(x_l-xdct_l_m))
m_l = max(abs(x_l-xdct_l_m))
idx_r = np.argmax(abs(x_r-xdct_r_m))
m_r = max(abs(x_r-xdct_r_m))
print('Max Error Left Channel = '+str(m_l))
print('Max Error Right Channel = '+str(m_r))