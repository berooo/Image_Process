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

def morthology(img,kernnel):
    #不全是白的，就设成黑的

    pass

def distanceTrans(img):

    pass

if __name__=='__main__':
    img=cv2.imread('bin_fingerPrint.jpg',0)
    print(img)
    kernel=np.ones([3,3])
    morthology(img,kernel)