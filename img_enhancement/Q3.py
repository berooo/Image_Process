#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q3.py
@time: 2019/11/1 19:24
@desc:
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

def flip(matrix):

    h,w=matrix.shape
    for i in range(h//2+1):
        for j in range(w):
            if i==h//2 and j>w//2:
                continue
            matrix[i][j],matrix[h-i-1][w-j-1]=matrix[h-i-1][w-j-1],matrix[i][j]

    return matrix

def twodConv(f,w,method='zero'):

    core_h,core_w=w.shape
    image_h,image_w=f.shape
    result_image=np.zeros([image_h,image_w])

    f_w=core_w//2
    f_h=core_h//2

    new_images=np.zeros([image_h+2*f_h,image_w+2*f_w])
    new_images[f_h:image_h+f_h,f_w:image_w+f_w]=f

    temp_core=flip(w)

    if method=='replicate':
        new_images[0:f_h,f_w:f_w+image_w]=f[0,:]
        new_images[image_h+f_h:,f_w:f_w+image_w]=f[image_h-1,:]
        new_images[f_h:f_h+image_h,0:f_w]=f[:,0].reshape([image_h,1])
        new_images[f_h:f_h+image_h,image_w+f_w:]=f[:,image_w-1].reshape([image_h,1])
        new_images[0:f_h,0:f_w]=f[0][0]
        new_images[image_h+f_h:,0:f_w]=f[image_h-1,0]
        new_images[0:f_h,image_w+f_w:]=f[0,image_w-1]
        new_images[image_h+f_h:,image_w+f_w:]=f[image_h-1,image_w-1]

    for i in range(image_h):
        for j in range(image_w):
            result_image[i][j] = np.sum(np.multiply(temp_core, new_images[i:i + core_h, j:j + core_w]))

    return result_image

def laplace(img):

    I=img
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    laplace_kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    h,w,d=I.shape

    for k in range(d):
        I[:,:,k]=twodConv(I[:,:,k],laplace_kernel,'replicate')

    plt.subplot(1,2,2)
    plt.imshow(I)
    plt.show()


if __name__=="__main__":
    img=cv2.imread('timg.jpg')
    img = img[:, :, (2, 1, 0)]
    laplace(img)
