#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q3.py
@time: 2019/10/1 16:53
@desc:
'''


import cv2
import numpy as np
import os
import convOp
import gaussKernel

def gaussian(f,sig,method='zero'):

    h,w,d=f.shape
    result_image=np.zeros([h,w,d])

    kernel = gaussKernel.gaussKernel(sig)
    for i in range(d):
        result_image[:,:,i]=convOp.twodConv(f[:,:,i],kernel,method)

    return result_image

if __name__=='__main__':

    for index,filename in enumerate(os.listdir('in')):

        f=cv2.imread('in\\'+filename)
        
        for i in range(1,6):
            if i==4:
                continue
            res=gaussian(f,i,method='replicate')
            cv2.imwrite('out\gaussian_'+filename.split('.')[0]+'_'+str(i)+'.jpg',res)

    img1=cv2.imread('in\cameraman.tif')
    res=gaussian(img1,5,method='zero')
    cv2.imwrite('out\zeropadding_cameraman.jpg',res)

    img2=cv2.imread('in\\newmandril.jpg')
    res = gaussian(img2, 5, method='zero')
    cv2.imwrite('out\zeropadding_newmandril.jpg',res)



