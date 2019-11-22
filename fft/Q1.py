#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q1.py
@time: 2019/10/12 14:27
@desc:
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def dft2D(f):

    f=np.fft.fft(f,axis=-1)
    f=np.fft.fft(f,axis=-2)

    return f

if __name__=='__main__':
    f=cv2.imread('rose512.tif',0)
    f=dft2D(f)
    fimg = np.log(np.abs(f))
    plt.imshow(fimg, 'gray')
    plt.show()
