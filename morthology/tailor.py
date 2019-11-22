#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: tailor.py
@time: 2019/11/11 15:08
@desc:
'''
import numpy as np
import cv2
import binaryzation
import thinning_Skeleton
import cordila_Skeleton
import matplotlib.pyplot as plt
import time

def union(imgA,imgB):
    h,w=imgA.shape

    for i in range(h):
        for j in range(w):
            if imgB[i][j]==255 and imgA[i][j]!=255:
                    imgA[i][j]=255

    return imgA

def intersection(imgA,imgB):
    h,w=imgA.shape
    newi=np.zeros([h,w])
    for i in range(h):
        for j in range(w):
            if imgA[i][j]==255 and imgB[i][j]==255:
                newi[i][j]=255

    return newi

def get_endPoints(img,kernels):

    newi=np.zeros(img.shape)
    for kernel in kernels:
        image,_=thinning_Skeleton.hit_or_miss(img,kernel)
        newi=union(newi,image)

    return newi

def once_thin(img,kernels):

    for kernel in kernels:
        # I=copy.deepcopy(img)
        hit_res, f = thinning_Skeleton.hit_or_miss(img, kernel)
        img = img - hit_res


    return img

def tailor(img,kernels):


    for i in range(3):
        img=once_thin(img,kernels)
        print(i)
    X1=img

    X2=get_endPoints(X1,kernels)
    H=np.ones([3,3])
    X3=intersection(cordila_Skeleton.inflate(X2,H),img)

    X4=union(X1,X3)

    return X4

if __name__=='__main__':

    start=time.clock()
    img=cv2.imread('in\\thin_zhiwen.jpg',0)
    img=binaryzation.binaryzation(img,100)

    plt.subplot(1,2,1)
    plt.imshow(img,'gray')

    kernel1=np.array([[-1,0,0],[1,1,0],[-1,0,0]])
    kernel2=np.array([[-1,1,-1],[0,1,0],[0,0,0]])
    kernel3=np.array([[0,0,-1],[0,1,1],[0,0,-1]])
    kernel4=np.array([[0,0,0],[0,1,0],[-1,1,-1]])

    kernel5=np.array([[1,0,0],[0,1,0],[0,0,0]])
    kernel6=np.array([[0,0,1],[0,1,0],[0,0,0]])
    kernel7=np.array([[0,0,0],[0,1,0],[0,0,1]])
    kernel8=np.array([[0,0,0],[0,1,0],[1,0,0]])

    kernels=np.asarray([kernel1,kernel2,kernel3,kernel4,kernel5,kernel6,kernel7,kernel8])

    newimg=tailor(img,kernels)

    plt.subplot(1,2,2)
    plt.imshow(newimg,'gray')

    plt.show()
    end=time.clock()
    print(end-start)