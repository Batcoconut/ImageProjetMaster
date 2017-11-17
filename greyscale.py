#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from math import *
from PIL import Image


n_neighbors = 2
height = 1024

"""
Created on Tue Oct 24 09:38:11 2017

@author: thibault
"""

LabelNotation = ["TR13" , "TR33" , "TR23" , "TR43" , "TR53" , "TR14" , "TR24" , "TR34" , "TR44" , "TR54", "TR15" , "TR25" , "TR45" , "TR55", "TR16" , "TR26" , "TR46" , "TR56","TREN"  ]


def ComputeDist(H_base,H_test):
    diff = np.zeros((H_base.shape[0],1))
    
    for i in range(0,H_base.shape[0]):
       
        diff[i] = np.linalg.norm(H_test-H_base[i])
   # print(diff)
    return diff

def solution_3():
    """
        CONSTRUCTION BASE APPRENTISSAGE

    """

    Liste = os.listdir('./BaseApprentissage')
    print(Liste)
    sz = len(Liste)
    Learn = np.zeros((sz,256))
    Label = np.zeros((sz,1))
    
    indice = 0;

    for i in range(0,len(Liste)):
        filename = './BaseApprentissage/' + Liste[i]
        #print(filename)

        Label[indice] = LabelNotation.index(Liste[i][0:4])
        img = cv2.imread(filename,0)
        Learn[indice] = (cv2.calcHist([img], [0], None, [256], [0, 256])).T
        indice = indice+1

    """
       CONSTRUCTION FEATURES TEST
    """
    indice = 0    
    Liste = os.listdir('./BaseTest')
    Test =np.zeros((len(Liste),sz))
    Test_label = np.zeros((len(Liste),1))
    for i in range(0,len(Liste)):
        filename = './BaseTest/' + Liste[i]    
        #print(filename)
        if(Liste[i] != ".DS_Store"):
                Test_label[indice] = LabelNotation.index(Liste[i][0:4])
                indice = indice+1
                img = cv2.imread(filename,0)
                hist = (cv2.calcHist([img], [0], None, [256], [0, 256])).T
                Test[i,:]= (ComputeDist(Learn,hist)).T
        
    """
        TEST BASE TEST KPPV
    """
    K = 5
    Label_algo = np.zeros((len(Liste),K))
    Dist_algo = np.zeros((len(Liste),K))
    for i in range(0,len(Liste)):
        for j in range(0,K):
            Label_algo[i,j] = Label[np.argmin(Test[i,:])]
            Dist_algo[i,j] = Test[i,np.argmin(Test[i,:])]
            Test[i,np.argmin(Test[i,:])] = float('inf')
    
    return Label_algo,Test_label,Dist_algo
"""
    APPELLE SOLUTION CHOISI
"""
Label_algo, Test_label,Dist_algo = solution_3()

Res = Label_algo - Test_label
suc1 = 0
suc2 = 0
suc3 = 0
suc4 = 0
suc5 = 0
for i in range(0,len(Res)):
        if Res[i,0] == 0:
            suc1+=1
        elif Res[i,1] == 0:
            suc2+=1
        elif Res[i,2] == 0:
            suc3+=1
        elif Res[i,3] == 0:
            suc4+=1
        elif Res[i,4] == 0:
            suc5+=1

print('Succès 1er: ',suc1, 'Succès 2eme: ', suc2, 'Succès 3eme: ', suc3, 'Succès 4eme: ', suc4, 'Succès 5eme: ', suc5)
print('Pourcentage réussite: ', suc1/len(Test_label)*100)

