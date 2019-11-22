#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q3.py
@time: 2019/10/12 16:57
@desc:
'''
import Q1
import Q2
import cv2
import matplotlib.pyplot as plt
import numpy as np

def normalization(f):
    return f/255.0

if __name__=='__main__':

    f=cv2.imread('rose512.tif',0)
    f=normalization(f)
    F=Q1.dft2D(f)
    g=Q2.idft2D(F)
    d=f-g
    fimg = np.abs(d)
    plt.imshow(fimg,'gray')
    plt.show()
