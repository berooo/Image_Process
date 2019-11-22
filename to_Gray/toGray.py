'''
@author:bero
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np


def rgb1gray(f,method='NTSC'):

    #cv2读取图片是b,g,r的顺序，改成r,g,b的顺序
    f=f[:,:,(2,1,0)]
    r,g,b=[f[:,:,int(x)] for x in range(3)]
    if method=='average':
        gray=(r+g+b)/3.0
    elif method=='NTSC':
        gray = r * 0.2989 + g * 0.5870 + b * 0.1140

    plt.imshow(gray,cmap="gray")
    plt.show()

    return gray

if __name__=='__main__':

    f=cv2.imread('lena512color.tiff')
    g=rgb1gray(f,'average')
    cv2.imwrite('newlena.jpg',g)

    f=cv2.imread('mandril_color.tif')
    g=rgb1gray(f)
    cv2.imwrite('newmandril.jpg',g)