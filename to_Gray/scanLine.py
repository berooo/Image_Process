'''
@author:bero
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np



def scanLine4e(f,I,loc):
    if loc=='row':
        newimg=f[I,:]
    elif loc=='col':
        newimg=f[:,I]
    plt.imshow(newimg)
    plt.show()


if __name__=='__main__':
    image1 = cv2.imread('cameraman.tif')
    h1=int(len(image1)/2)
    scanLine4e(image1,h1,'row')

    image2=cv2.imread('einstein.tif')
    w2=int(len(image2[0])/2)
    scanLine4e(image2,w2,'col')