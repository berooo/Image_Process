#!/usr/bin/env python
# encoding: utf-8
'''
@author: shibaorong
@license: (C) Copyright 2019, Node Supply Chain Manager Corporation Limited.
@contact: diamond_br@163.com
@software: pycharm
@file: Q2.py
@time: 2019/10/1 15:28
@desc:
'''
import math
import numpy as np

def gaussKernel(sig,m=None):

    std_m=math.ceil(sig*3)*2+1

    #m没有提供的情况
    if m==None:
        m=std_m

    if m<std_m:
        print('\033[0;31mwarnings: m is too small!\033[0m')

    kernel=np.zeros([m,m])
    k = m // 2

    for i in range(m):
        for j in range(m):
            kernel[i][j]=(1/(2*np.pi*pow(sig,2)))*np.exp(-(pow(i-k,2)+pow(j-k,2))/(2*pow(sig,2)))


    kernel=kernel/np.sum(kernel)

    return kernel

if __name__=='__main__':
    print(gaussKernel(0.1,3))
    print(gaussKernel(1,3))
