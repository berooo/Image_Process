#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q2.py
@time: 2019/10/12 16:17
@desc:
'''
import Q1
import numpy as np
import cv2
import matplotlib.pyplot as plt

def idft2D(F):

    F_shape=F.shape
    F=F.conj()
    F=np.fft.fft(F,axis=-2)
    F = np.fft.fft(F, axis=-1)
    F/=F_shape[0]*F_shape[1]
    F=F.conj()
    return F

if __name__=='__main__':
    F = cv2.imread('rose512.tif', 0)
    #plt.imshow(F/255.0, 'gray')
    F=Q1.dft2D(F)
    C=idft2D(F)
    fimg = np.abs(C)
    plt.imshow(fimg, 'gray')
    plt.show()
