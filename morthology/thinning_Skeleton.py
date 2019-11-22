#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: tailoring.py
@time: 2019/11/10 23:30
@desc:
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import binaryzation
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

def match(kernel,b):
    h,w=kernel.shape
    result=np.zeros(kernel.shape)

    for i in range(h):
        for j in range(w):
         if kernel[i][j]==-1:
             result[i][j]=0
         else:
             result[i][j]=int(np.logical_xor(kernel[i][j],b[i][j]))

    is_match=np.sum(result)==0

    return is_match

def hit_or_miss(img,kernel):
    h,w=img.shape
    core_h,core_w=kernel.shape
    img=padding(img,kernel)
    temp=np.zeros(img.shape)

    f_h=core_h//2
    f_w=core_w//2

    flag=0
    for i in range(h):
        for j in range(w):
            cube=img[i:i+core_h,j:j+core_w]
            if match(kernel,cube):
                temp[i+1][j+1]=255
                flag=1
            else:
                temp[i+1][j+1]=0

    img=temp[1:-1,1:-1]

    return img,flag

def thinning(img,kernels):

    flag=-1
    while flag!=0:
        flag=0
        for kernel in kernels:
            #I=copy.deepcopy(img)
            hit_res,f=hit_or_miss(img,kernel)
            flag+=f
            img = img - hit_res
            plt.imshow(img,'gray')
            plt.show()

    return img


if __name__=='__main__':
    start=time.clock()

    img=cv2.imread('in\\gujia.png',0)
    img=binaryzation.binaryzation(img,100)



    kernel1 = np.array([[0, 0, 0], [-1, 1, -1], [1, 1, 1]])
    kernel3 = np.array([[1, -1, 0], [1, 1, 0], [1, -1, 0]])
    kernel5 = np.array([[1, 1, 1], [-1, 1, -1], [0, 0, 0]])
    kernel7 = np.array([[0, -1, 1], [0, 1, 1], [0, -1, 1]])

    kernel2 = np.array([[-1, 0, 0], [1, 1, 0], [1, 1, -1]])
    kernel4 = np.array([[1, 1, -1], [1, 1, 0], [-1, 0, 0]])
    kernel6 = np.array([[-1, 1, 1], [0, 1, 1], [0, 0, -1]])
    kernel8 = np.array([[0, 0, -1], [0, 1, 1], [-1, 1, 1]])

    kernels = [kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8]

    newimg = thinning(img, kernels)

    plt.subplot(1, 2, 1)
    plt.imshow(img, 'gray')
    plt.subplot(1,2,2)
    plt.imshow(newimg,'gray')

    plt.show()

    end=time.clock()
    print(end-start)

    cv2.imwrite('in\\thin_gujia.jpg',newimg)

