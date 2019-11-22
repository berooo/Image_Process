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


import cv2
import matplotlib.pyplot as plt

def binaryzation(img,T):

    if (img[0][0] > 250):
        img = 255 - img

    w,h=img.shape
    for i in range(w):
        for j in range(h):
            if img[i][j]<T:
                img[i][j]=0
            else:
                img[i][j]=255

    return img


if __name__=="__main__":

    img=cv2.imread('in\\fingerPrint.jpg',0)
    img=binaryzation(img,150)
    plt.imshow(img,'gray')
    plt.show()