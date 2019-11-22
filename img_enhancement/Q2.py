#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q2.py
@time: 2019/11/1 19:24
@desc:
'''
import sys
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def add_noise(img,percentage):

    NoiseNum=int(percentage*img.shape[0]*img.shape[1])

    for i in range(NoiseNum):
        rand_x=random.randint(0,img.shape[0]-1)
        rand_y=random.randint(0,img.shape[1]-1)

        if random.random()<0.5:
            img[rand_x][rand_y]=0

        else:
            img[rand_x][rand_y]=255


def smoothing(img):
    h,w,d=img.shape

    for i in range(h-4):
        for j in range(w-4):
            for k in range(d):
                img[i+2][j+2][k]=kernel_processing(img[:,:,k], i, j)

    return img


def kernel_processing(img,i,j):
    m=i+2
    n=j+2
    mask1=img[i+1:i+4,j+1:j+4]

    mask2=np.array([img[m][n],img[m-1][n-1],img[m-1][n],img[m-1][n+1],img[m-2][n-1],img[m-2][n],img[m-2][n+1]])
    mask3=np.array([img[m][n],img[m-1][n-1],img[m][n-1],img[m+1][n-1],img[m-1][n-2],img[m][n-2],img[m+1][n-2]])
    mask4=np.array([img[m][n],img[m+1][n-1],img[m+1][n],img[m+1][n+1],img[m+2][n-1],img[m+2][n],img[m+2][n+1]])
    mask5=np.array([img[m][n],img[m-1][n+1],img[m][n+1],img[m+1][n+1],img[m-1][n+2],img[m][n+2],img[m+1][n+2]])

    mask6=np.array([img[m][n],img[m-1][n],img[m][n-1],img[m-2][n-1],img[m-1][n-2],img[m-2][n-2],img[m-1][m-1]])
    mask7=np.array([img[m][n],img[m][n+1],img[m+1][n],img[m+1][n+1],img[m+1][n+2],img[m+2][n+1],img[m+2][n+2]])
    mask8 = np.array([img[m][n],img[m][n+1],img[m-1][n],img[m-1][n+1],img[m-1][n+2],img[m-2][n+1],img[m-2][n+2]])
    mask9 = np.array([img[m][n],img[m][n-1],img[m+1][n],img[m+1][n-1],img[m+1][n-2],img[m+2][n-1],img[m+2][n-2]])

    list=[mask1,mask2,mask3,mask4,mask5,mask6,mask7,mask8,mask9]

    index=-1
    maxvalue=sys.maxsize
    for i,j in enumerate(list):
        temp=np.var(j)
        if temp<maxvalue:
            maxvalue=temp
            index=i


    return np.mean(list[index])


if __name__=="__main__":
    img=cv2.imread('lena512color.tiff')
    img = img[:, :, (2, 1, 0)]
    add_noise(img,0.05)
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    new_img=smoothing(img)
    plt.subplot(1, 2, 2)
    plt.imshow(new_img)
    plt.show()
