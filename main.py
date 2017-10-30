#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

NombrePoints = int(input('Entrer le nombre de point par bloc souhaité'))
NombreValeurs_descripteur = NombrePoints*32
n_neighbors = 5

"""
Created on Tue Oct 24 09:38:11 2017

@author: thibault
"""

LabelNotation = ["TR56" , "TR66", "TR65","TR33","TR34","TR35","TR36"]



def Image_blocCompute(image_filename):
    img = cv2.imread(image_filename,0)
    taille = img.shape
    X_crop = int(taille[0]/3)
    Y_crop = int(taille[1]/3)
    
    Liste_ORB_crop = []
    for i in range(0,3):
        for j in range(0,3):
            crop_img = img[i*X_crop:i*X_crop + X_crop , j*Y_crop:j*Y_crop + Y_crop]
            #calcul ORB de l'image crop
            crop_orb, crop_orb_pts = ORB(crop_img)
            if(crop_orb != None):
                if(len(crop_orb_pts) > NombrePoints):
                    
                    #recuperation de NombrePoints
                    Select_Features = crop_orb[0:NombrePoints , 0:32]
                    val = Select_Features.flat[:]
                    
                    #integration dans le tableau des Learn
                    Liste_ORB_crop.append(val)

    return Liste_ORB_crop


    
def ComputeFeatures(image_filename):
    img = cv2.imread(image_filename,0)
    return ORB(img)

def ORB(img):
    ##Detection des surf points
    orb = cv2.ORB_create()
    pts , Features = orb.detectAndCompute(img,None)
    return Features , pts


def solution_1():
    
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
        val,pts = ComputeFeatures(filename)
        Learn[indice] = val.flat[:]
    
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
        val , pts = ComputeFeatures(filename)
        Test[indice] = val.flat[:]
        indice = indice+1
    
    
    clf = NearestCentroid() 
    clf.fit(Learn,Label)
    
    
    """
        TEST BASE TEST
    """
    
    
    indice = clf.predict(Test)
    print(indice)
    
    
    for i in range(0,len(indice)):
        print('image test : ' , Liste[i])
        print('label res : ' , LabelNotation[int(indice[i])])


def solution_2():
    
    """
        CONSTRUCTION BASE APPRENTISSAGE
    
    """
    
    Liste = os.listdir('./BaseApprentissage')
    print(Liste)

    Learn_tempo = []
    Label_tempo = []

    for i in range(0,len(Liste)):
        filename = './BaseApprentissage/' + Liste[i]
        print(filename)
        
        Image_crop = Image_blocCompute(filename)
        print('nombe de image crop correct : ' , len(Image_crop) )
        
        for val in Image_crop:
            Learn_tempo.append(val)
            Label_tempo.append(LabelNotation.index(Liste[i][0:4]))
            
    
    print('Label et Learn construit Taille : ' , len(Learn_tempo), len(Label_tempo))
    
    #construction de Learn et Label
    Learn = np.zeros((len(Learn_tempo),NombreValeurs_descripteur))
    Label = np.zeros((len(Learn_tempo),1))
    for i in range(0,len(Learn_tempo)):
        Learn[i] = Learn_tempo[i]
        Label[i] = Label_tempo[i]
    
    clf = KNeighborsClassifier(n_neighbors) 
    clf.fit(Learn,Label)
    
    """
        CONSTRUCTION FEATURES TEST
    """
    
    Liste = os.listdir('./BaseTest')
    Test_tempo = []
    Test_label = np.zeros((len(Liste),1))
    Image = []
    indice = 0
    for i in range(0,len(Liste)):
        filename = './BaseTest/' + Liste[i]
        print(filename)
        
        Test_label[indice] = LabelNotation.index(Liste[i][0:4])
        indice = indice+1
        
        Image_crop = Image_blocCompute(filename)
        for val in Image_crop:
            Image.append(i)
            Test_tempo.append(val)
            
    #construction de Test
    Test = np.zeros((len(Test_tempo),NombreValeurs_descripteur))
    for i in range(0,len(Test_tempo)):
        Test[i] = Test_tempo[i]
    
    """
        TEST BASE TEST
    """
    
    res = clf.predict(Test)

    for i in range(0,len(Liste)):
        tab = []
        for j in range(0,len(res)):
            if(Image[j] == i):
                tab.append(int(res[j]))
        
        #trouve la valeur en plus grand nombre
        indiceMax =0
        for k in range(0,len(LabelNotation)):
            if(tab.count(k) > tab.count(indiceMax)):
                indiceMax=k
        
        print('resultat image :' ,Liste[i])
        print('Valeur la plus présente : ' , indiceMax, 'tour ' ,LabelNotation[indiceMax], ' pourcentage :'  ,int(tab.count(indiceMax)/len(tab) *100) , '%')
        if Liste[i][0:4] == LabelNotation[indiceMax]:
            print('')
            print('Find success !!!')
            print('')
        else:
            print('')
            print('Find error !!!')
            print('')
            


"""
    APPELLE SOLUTION CHOISI
"""
solution_2()