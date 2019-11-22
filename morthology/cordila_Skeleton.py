#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: morthology_Skeleton.py
@time: 2019/11/10 10:44
@desc:
'''
import numpy as np
import cv2
import binaryzation
import matplotlib.pyplot as plt
import copy
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

def inflate(img,kernel):
    core_h, core_w = kernel.shape
    img_h, img_w = img.shape
    newimg = padding(img, kernel)

    for i in range(img_h):
        for j in range(img_w):
            if img[i][j]*kernel[1][1]!=0:
                newimg[i:i+core_h,j:j+core_w]=255

    f_w = core_w // 2
    f_h = core_h // 2
    img=newimg[f_h:img_h+f_h,f_w:img_w+f_w]
    return img

def open_op(img,kernel):

    img=corrotion(img,kernel)
    img=inflate(img,kernel)
    return img

def subtract(img1,img2):
    h,w=img1.shape
    newimg=np.zeros(img1.shape)
    for i in range(h):
        for j in range(w):
            newimg[i][j]=img1[i][j]-img2[i][j]
            if(img1[i][j]<img2[i][j]):
                print(1)
    return newimg

def morthology_ske(img,kernel):
    L=[]
    len=-1
    img=img.astype(np.float)

    while len!=0:

        I1=copy.deepcopy(img)
        I2=copy.deepcopy(img)

        f=open_op(I1,kernel)
        s=subtract(I2,f)
        L.append(s)
        img=corrotion(img,kernel)
        len=np.sum(img)

    newimg=np.zeros(img.shape)
    for i in L:
        newimg=cv2.bitwise_or(i,newimg)

    return newimg

if __name__=='__main__':
    start = time.clock()

    img=cv2.imread('in\\fingerPrint.jpg',0)
    img=binaryzation.binaryzation(img,100)

    plt.subplot(1,2,1)
    plt.imshow(img, 'gray')

    kernel=np.ones([3,3])
    new_img=morthology_ske(img,kernel)

    plt.subplot(1,2,2)
    plt.imshow(new_img,'gray')

    plt.show()
    end = time.clock()

    print(end - start)

