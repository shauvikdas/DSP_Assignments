# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 00:22:22 2018

@author: Shauvik
"""

from __future__ import division
import numpy as np
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

[rate,data] = scipy.io.wavfile.read("song.wav", mmap=False)

t_start = 0
t_end = (t_start + 1)

N = 44100
start_index = t_start * N
end_index = t_end * N
signal = data[start_index:end_index]
l_channel = signal[:,0]
r_channel = signal[:,1]


x_l = l_channel
x_r = r_channel

XDCT_L = scipy.fftpack.dct(x_l, norm = 'ortho')
XDCT_R = scipy.fftpack.dct(x_r, norm = 'ortho')

M = 2000
XDCT_L_m = []
XDCT_R_m = []

for k in range (N-M):
    XDCT_L_m.append(XDCT_L[k])
    XDCT_R_m.append(XDCT_R[k])
        
for k in range (N-M,N):
    XDCT_L_m.append(0)
    XDCT_R_m.append(0)
    
X = [np.array([N,M])]
for i in range(N-M):
    a_ = np.array([XDCT_L_m[i],XDCT_R_m[i]])
    X.append(a_)
X = np.array(X)
np.savetxt('compressedFile.txt',X)