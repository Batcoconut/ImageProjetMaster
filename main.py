#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import os


"""
Created on Tue Oct 24 09:38:11 2017

@author: thibault
"""

LabelNotation = ["TR56" , "TR66"]

def ComputeFeatures(image_filename):
    img = cv2.imread(filename,0)
    ##Detection des surf points
    orb = cv2.ORB_create()
    pts , Features = orb.detectAndCompute(img,None)
    test = Features.flat[:]
    return test


"""
    CONSTRUCTION BASE APPRENTISSAGE

"""

Liste = os.listdir('./BaseApprentissage')
print(Liste)
Learn = np.zeros((len(Liste),16000))
Label = np.zeros((len(Liste),1))

indice = 0; 

for i in range(0,len(Liste)):
    filename = './BaseApprentissage/' + Liste[i]
    print(filename)
    
    Label[indice] = LabelNotation.index(Liste[i][0:4])
    Learn[indice] = ComputeFeatures(filename)
    
    indice = indice+1


"""
    CONSTRUCTION FEATURES TEST
"""

Liste = os.listdir('./BaseTest')
Test = np.zeros((len(Liste),16000))
indice = 0

for i in range(0,len(Liste)):
    filename = './BaseTest/' + Liste[i]
    print(filename)
    Test[indice] = ComputeFeatures(filename)
    indice = indice+1





clf = NearestCentroid()
clf.fit(Learn,Label)



"""
    TEST BASE TEST
"""

indice = clf.predict(Test)
print(indice)

for i in range(0,len(indice)):
    print(LabelNotation[int(indice[i])])

    