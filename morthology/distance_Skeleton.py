#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Skeleton_extraction.py
@time: 2019/11/9 20:52
@desc:
'''

import numpy as np
import cv2
import copy
import binaryzation
import matplotlib.pyplot as plt
import time

def padding(f,w):
    core_h, core_w = w.shape
    image_h, image_w = f.shape
    result_image = np.zeros([image_h, image_w])

    f_w = core_w // 2
    f_h = core_h // 2

    new_images = np.zeros([image_h + 2 * f_h, image_w + 2 * f_w])
    new_images[f_h:image_h + f_h, f_w:image_w + f_w] = f

    return new_images

def corrotion(img,kernel):
    core_h,core_w=kernel.shape
    img_h,img_w=img.shape
    newimg=padding(img,kernel)

    for i in range(img_h):
        for j in range(img_w):
            t=np.sum(np.multiply(kernel,newimg[i:i + core_h, j:j + core_w]))
            if t<255*core_w*core_h:
                img[i][j]=0
            else:
                img[i][j]=255

    return img

def getBoundary(img,kernel):

    I=copy.deepcopy(img)
    imgi=corrotion(I,kernel)
    res_img=img-imgi

    return res_img

def dis_ske(img):

    h,w=img.shape
    kernel=np.ones([3,3])
    img=padding(img,kernel)

    disimg=getBoundary(img,kernel)

    dis=img-disimg

    distance=copy.deepcopy(dis)

    for i in range(h):
        for j in range(w):
            temp0=distance[i][j]+4
            temp1=min(temp0,distance[i+1][j]+3)
            temp2=min(temp1,distance[i][j+1]+3)
            temp3=min(temp2,distance[i][j+2]+4)
            distance[i+1][j+1]=min(temp3,distance[i+1][j+1])



    for i in range(h-1,-1,-1):
        for j in range(w-1,-1,-1):
            temp0=distance[i+2][j+2]+4
            temp1=min(temp0,distance[i+2][j+1]+3)
            temp2=min(temp1,distance[i+1][j+2]+3)
            temp3=min(temp2,distance[i+2][j]+4)
            distance[i+1][j+1]=min(temp3,distance[i+1][j+1])

    return distance




if __name__=='__main__':

    start=time.clock()

    img=cv2.imread('in\\fingerPrint.jpg',0)
    img=binaryzation.binaryzation(img,100)
    plt.subplot(1,2,1)
    plt.imshow(img,'gray')

    skeleton=dis_ske(img)
    plt.subplot(1,2,2)
    plt.imshow(skeleton, 'gray')
    plt.show()

    end=time.clock()

    print(end-start)