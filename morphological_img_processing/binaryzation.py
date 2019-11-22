#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: binaryzation.py
@time: 2019/11/8 19:49
@desc:
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

def binaryzation(img,T):

    w,h=img.shape
    for i in range(w):
        for j in range(h):
            if img[i][j]<T:
                img[i][j]=255
            else:
                img[i][j]=0

    plt.imshow(img,'gray')
    plt.show()

    return img




if __name__=="__main__":

    img=cv2.imread('fingerPrint.jpg',0)
    img=binaryzation(img,150)
    cv2.imwrite('bin_fingerPrint.jpg', img)