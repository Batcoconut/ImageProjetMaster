#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:04:14 2017

@author: alexis
"""

import cv2
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt


print('list des images')
Liste = os.listdir('Image_data')

img = cv2.imread('Image_data/' +Liste[1],0)
#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.subplot(211), plt.plot(hist)
plt.subplot(212), plt.imshow(img,'gray')
plt.show()


# =============================================================================
#for i in range (1,len(Liste)):
#    print(Liste[i])
#   img = cv2.imread('Image_data/' +Liste[i])    
#   cv2.imshow(img,)
#
#     
#     h = np.zeros((50,256))
#     bins = np.arange(32).reshape(32,1)
#     hist_item = cv2.calcHist([img],0,None,[32],[0,256])
#     cv2.normalize(hist_item,hist_item,64,cv2.NORM_MINMAX)
#     hist=np.int32(np.around(hist_item))
#     pts = np.column_stack((bins,hist))
#     cv2.polylines(h,[pts],False,(255,255,255))
#     
#     h=np.flipud(h)
#     
#     cv2.imshow('colorhist',h)
# =============================================================================