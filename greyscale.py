#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:04:14 2017

@author: alexis
"""

import cv2
import numpy as np
import os
from scipy.spatial import distance
from PIL import Image
from matplotlib import pyplot as plt


def ComputeDist(H_base,H_test,label):
    diff = np.zeros((H_base.shape[0],1))
    
    for i in range(0,H_base.shape[0]):
        diff[i] = np.linalg.norm(H_test-H_base[i])
    print(diff)
    return diff#label[np.argmin(diff)]

print('list des images')
Liste = os.listdir('testHist/Base')
hist = np.zeros((len(Liste),256))
lab = []

#========= Base
for i in range (0,len(Liste)):
    img = cv2.imread('testHist/Base/' +Liste[i],0)
    hist[i,:] = (cv2.calcHist([img], [0], None, [256], [0, 256])).T
    lab.insert(1,Liste[i])
    plt.plot(np.linspace(0,1,256),hist[i,0:256].T, label=Liste[i])
  #  hist[i,end] = Liste[i]
#    plt.subplot(211), plt.plot(hist)
 #   plt.subplot(212), plt.imshow(img,'gray')
 #   plt.show()

# Test
  
img = cv2.imread('testHist/test.jpg',0)
  
h_test = (cv2.calcHist([img], [0], None, [256], [0, 256])).T

plt.plot(np.linspace(0,1,256),h_test.T,label='test')
plt.legend()
plt.show()

print(ComputeDist(hist,h_test,lab))
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


def solution_3():
    
    """
        RESIZE DE TOUTE LES IMAGES
    """
    choix = int(input('resize image 1 oui / 2 non'))
    if(choix == 1):
        ResizeImage()

    """
        CONSTRUCTION BASE APPRENTISSAGE

    """

    Liste = os.listdir('./BaseApprentissage')
    print(Liste)
    Learn = np.zeros((len(Liste),16000))
    Label = np.zeros((len(Liste),1))

    indice = 0;

    for i in range(1,len(Liste)):
        filename = './BaseApprentissage/' + Liste[i]
        print(filename)

        Label[indice] = LabelNotation.index(Liste[i][0:4])
        val,pts = ComputeFeatures(filename)
        Learn[indice] = val.flat[:]

        indice = indice+1

    
    
    
    
    
    