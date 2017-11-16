#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from collections import Counter
from sklearn.neighbors.nearest_centroid import NearestCentroid
import os
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from math import *
from PIL import Image




n_neighbors = 1
height = 1024



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

def ORB(img):
    ##Detection des surf points
    orb = cv2.ORB_create()
    pts , Features = orb.detectAndCompute(img,None)
    return Features , pts

def Image_blocCompute(image_filename):
    img = cv2.imread(image_filename,0)
    Liste_ORB_crop = []
    test = True
    
    
    #on applique direct sur l'image entière
    crop_orb, crop_orb_pts = ORB(img)
    if(crop_orb != None):
        nombre_points = len(crop_orb)
        nombre_points = 200
        for k in range(0,nombre_points):
            val = crop_orb[k].astype(int)
            if(test):
                Liste_ORB_crop = val
                test =False
            else:
                Liste_ORB_crop = np.vstack((Liste_ORB_crop,val))

    return Liste_ORB_crop


def compteur(tab,indice):
    val = 0;
    for i in range(0,len(tab)):
        if(tab[i] == indice):
            val = val+1
    return val

def main():
    """
        RESIZE DE TOUTE LES IMAGES
    """

    choix = int(input('resize image 1 oui / 2 non : '))

    if(choix == 1):
        ResizeImage()
    else:
        print('No resize!')
    
    """
        CONSTRUCTION BASE APPRENTISSAGE

    """

    Liste = os.listdir('./BaseApprentissage')
    NombreImage = len(Liste) 

    Learn = []
    Label = []


    for i in range(0,NombreImage):
        filename = './BaseApprentissage/' + Liste[i]
        print(filename)
        if(Liste[i] != ".DS_Store"):
            Features_tab = Image_blocCompute(filename)
            #renvoi un tableau de points qu'il faut labelise
            if (i==0):
                Learn = Features_tab
                for k in range(0,len(Features_tab)):
                    Label.append(LabelNotation.index(Liste[i][0:4]))

            else:
                Learn = np.vstack((Learn,Features_tab))
                for k in range(0,len(Features_tab)):
                    Label.append(LabelNotation.index(Liste[i][0:4]))

        

    print(Learn)
    print('taille Learn : ' , Learn.shape)
    print('taille Label' , len(Label))


    
    #Apprentissage a partir de la base de donnée d'apprentissage
    clf = KNeighborsClassifier(n_neighbors)

    print('Nombre de points apprentissage : ' , len(Learn))



    clf.fit(Learn,Label)
    
    #Construction Base test
    
    
    Liste = os.listdir('./BaseTest')
    NombreImage = len(Liste) 
    sucess = 0;
    nb_test =0;


    for i in range(0,len(Liste)):
        filename = './BaseTest/' + Liste[i]
        if(Liste[i] !=".DS_Store"):
            nb_test =nb_test+1
            Features_test = Image_blocCompute(filename)
            res = clf.predict(Features_test)
            #trouve la valeur en plus grand nombre
            indiceMax = 0
            valeurIndiceMax =0;
            for j in range(0,len(LabelNotation)):
                cpt = compteur(res,j)
                if(cpt>valeurIndiceMax):
                    valeurIndiceMax=cpt
                    indiceMax =j
            
            print('image : ',Liste[i] , ' tour find : ' , LabelNotation[indiceMax] , ' ratio val trouve :' , valeurIndiceMax,'/',len(res))
            print(' label res premièer image : ' , LabelNotation[res[0]] , LabelNotation[res[1]] , LabelNotation[res[1]] , LabelNotation[res[1]] )
            if Liste[i][0:4] == LabelNotation[indiceMax]:
                sucess = sucess+1
    
    return  sucess , nb_test

    

suc, res = main()
print(' success : ' , suc , ' / nb test : ' , res , ' / % : ' , (suc/res)*100)