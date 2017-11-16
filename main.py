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

def ResizeImage():

    FolderTest = './BaseTest/'
    FolderApp = './BaseApprentissage/'

    Liste = os.listdir(FolderApp)
    for i in range (0,len(Liste)):
        print(Liste[i])
        if(Liste[i] != '.DS_Store'):
            img = Image.open (FolderApp +Liste[i])
            wpercent = (height / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            newImage = img.resize((height, hsize))
            print(newImage)
            newImage.save(FolderApp + Liste[i])

    Liste = os.listdir(FolderTest)
    for i in range (0,len(Liste)):

        if(Liste[i] != '.DS_Store'):
            img = Image.open (FolderTest +Liste[i])
            wpercent = (height / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            newImage = img.resize((height, hsize))
            print(newImage)
            newImage.save(FolderTest + Liste[i])

def Image_blocCompute(image_filename,NombrePoints):
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
    orb = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=200)

    pts , Features = orb.detectAndCompute(img,None)
    return Features , pts

def solution_1():
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
    Learn = np.zeros((len(Liste),200))
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
    Test = np.zeros((len(Liste),200))
    indice = 0

    for i in range(0,len(Liste)):
        filename = './BaseTest/' + Liste[i]
        print(filename)
        val , pts = ComputeFeatures(filename)
        Test[indice] = val.flat[:]
        indice = indice+1


    clf = KNeighborsClassifier(n_neighbors)
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
        RESIZE DE TOUTE LES IMAGES
    """
    choix = int(input('resize image 1 oui / 2 non : '))
    if(choix == 1):
        ResizeImage()

    n_neighbors = int(input('Valeur de K : '))
    NombrePoints = int(input('Entrer le nombre de point par bloc souhaité : '))
    NombreValeurs_descripteur = NombrePoints*32


    """
        CONSTRUCTION BASE APPRENTISSAGE

    """

    Liste = os.listdir('./BaseApprentissage')
    print(Liste)

    Learn_tempo = []
    Label_tempo = []

    for i in range(1,len(Liste)):
        filename = './BaseApprentissage/' + Liste[i]
        print(filename)
        if(Liste[i] != ".DS_Store"):
            Image_crop = Image_blocCompute(filename,NombrePoints)
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
    for i in range(1,len(Liste)):
        filename = './BaseTest/' + Liste[i]
        print(filename)
        if(Liste[i] != ".DS_Store"):
            Test_label[indice] = LabelNotation.index(Liste[i][0:4])
            indice = indice+1

            Image_crop = Image_blocCompute(filename,NombrePoints)
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

    success = 0
    nb_test = 0

    for i in range(0,len(Liste)):
        tab = []
        if(Liste[i] != ".DS_Store"):
            nb_test = nb_test+1
            for j in range(0,len(res)):
                if(Image[j] == i):
                    tab.append(int(res[j]))

            #trouve la valeur en plus grand nombre
            indiceMax =0
            for k in range(0,len(LabelNotation)):
                if(tab.count(k) > tab.count(indiceMax)):
                    indiceMax=k

            print('resultat image :' ,Liste[i])
            if tab.count(indiceMax) != 0:
                print('Valeur la plus présente : ' , indiceMax, 'tour ' ,LabelNotation[indiceMax], ' pourcentage :'  ,int(tab.count(indiceMax)/len(tab) *100) , '%')
            else:
                print('Valeur la plus présente : ' , indiceMax, 'tour ' ,LabelNotation[indiceMax])
            if Liste[i][0:4] == LabelNotation[indiceMax]:
                success = success+1;
                print('success!')

    return success, nb_test

"""
    APPELLE SOLUTION CHOISI
"""
suc , res =solution_1()
print('nombre test : ' , res , ' nombre succes : ' , suc ,  ' % : ' , (suc/res)*100)
