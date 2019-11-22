#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q1.py
@time: 2019/11/1 19:24
@desc:
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def histequal4e(I):

    h=I.shape[0]
    img=I.flatten()
    img_len=len(img)
    img_int=img.astype(np.int)

    origin_hist=np.zeros(256)
    for i in range(img_len):
        origin_hist[img_int[i]]+=1
    origin_hist=[origin_hist[i]/img_len for i in range(256)]

    new_hist=np.zeros(256)
    new_hist[0]=origin_hist[0]
    for i in range(255):
        new_hist[i+1]=origin_hist[i+1]+new_hist[i]

    new_img=np.array([255*new_hist[img_int[i]] for i in range(len(img_int))])
    new_img=new_img.reshape([h,-1])

    plt.subplot(1,2,1)
    plt.imshow(I,'gray')
    plt.subplot(1,2,2)
    plt.imshow(new_img,'gray')

    plt.show()

if __name__=="__main__":
    img=cv2.imread('test.jpeg',0)
    print(img.shape)
    histequal4e(img)