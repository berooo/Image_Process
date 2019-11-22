#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q4.py
@time: 2019/10/12 19:21
@desc:
'''
import numpy as np
import matplotlib.pyplot as plt
import Q1

def generateImg():
    f=np.zeros([512,512],dtype=np.float)
    f[225:285,250:260]=1
    return f


if __name__=='__main__':
    plt.figure()
    #产生图像
    f=generateImg()
    plt.subplot(2,2,1)
    plt.imshow(f,'gray')

    #fft变换
    f_fft=Q1.dft2D(f)
    plt.subplot(2,2,2)
    plt.imshow(np.abs(f_fft),'gray')

    #中心化
    F=np.fft.fftshift(f_fft)
    plt.subplot(2,2,3)
    plt.imshow(np.abs(F),'gray')

    #对数变换
    s=np.log(1+np.abs(F))
    plt.subplot(2,2,4)
    plt.imshow(s,'gray')

    plt.show()