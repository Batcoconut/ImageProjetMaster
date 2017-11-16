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
    process = cv2.ORB_create(edgeThreshold=15, patchSize=31, nlevels=8, fastThreshold=20, scaleFactor=1.2, WTA_K=2,scoreType=cv2.ORB_HARRIS_SCORE, firstLevel=0, nfeatures=50)
    #process = cv2.xfeatures2d.SIFT_create()

    pts , Features = process.detectAndCompute(img,None)
    return Features , pts

def Image_blocCompute(image_filename):
    img = cv2.imread(image_filename,0)
    Liste_ORB_crop = []
    test = True

    
    taille = img.shape
    decoupage = 2
    
    X_crop = int(taille[0]/decoupage)
    Y_crop = int(taille[1]/decoupage)


    for i in range(0,decoupage):
        for j in range(0,decoupage):
            crop_img = img[i*X_crop:i*X_crop + X_crop , j*Y_crop:j*Y_crop + Y_crop]
            #calcul ORB de l'image crop
            crop_orb =[]
            crop_orb, crop_orb_pts = ORB(crop_img)
            if(len(crop_orb) != 0):
                nombre_points = len(crop_orb)

                for k in range(0,nombre_points):
                    val = crop_orb[k].astype(int)
                    if(test):
                        Liste_ORB_crop = val
                        test =False
                    else:
                        Liste_ORB_crop = np.vstack((Liste_ORB_crop,val))
    return Liste_ORB_crop


def Image_blocCompute_Predict(image_filename):
    img = cv2.imread(image_filename,0)
    Liste_ORB_crop_tab = []
    

    
    taille = img.shape
    decoupage = 2
    X_crop = int(taille[0]/decoupage)
    Y_crop = int(taille[1]/decoupage)


    for i in range(0,decoupage):
        for j in range(0,decoupage):
            crop_img = img[i*X_crop:i*X_crop + X_crop , j*Y_crop:j*Y_crop + Y_crop]
            #calcul ORB de l'image crop
            crop_orb = []
            crop_orb, crop_orb_pts = ORB(crop_img)
            Liste_ORB_crop = []
            test = True
            
            if(len(crop_orb) != 0):
                nombre_points = len(crop_orb)

                for k in range(0,nombre_points):
                    val = crop_orb[k].astype(int)
                    if(test):
                        Liste_ORB_crop = val
                        test =False
                    else:
                        Liste_ORB_crop = np.vstack((Liste_ORB_crop,val))

                Liste_ORB_crop_tab.append(Liste_ORB_crop)

    return Liste_ORB_crop_tab


def compteur(tab,indice):
    val = 0;
    for i in range(0,len(tab)):
        if(tab[i] == indice):
            val = val+1
    return val

def compteurPoids(tab,indice):

    val = 0;
    for i in range(0,len(tab)):
        if(tab[i] == indice):
            if( i <= 5):
                val = val+4
            elif (i > 5 and i< 20):
                val = val+2
            else:
                val = val+1


    return val


def ComputeDist(H_base,H_test):
    diff = np.zeros((H_base.shape[0],1))
    
    for i in range(0,H_base.shape[0]):
        diff[i] = np.linalg.norm(H_test-H_base[i])
   # print(diff)
    return diff

def Histoprocess():
    
    """
        CONSTRUCTION BASE APPRENTISSAGE

    """

    Liste = os.listdir('./BaseApprentissage')
    sz = len(Liste)
    Learn = np.zeros((sz,256))
    Label = np.zeros((sz,1))
    
    indice = 0;

    for i in range(0,len(Liste)):
        filename = './BaseApprentissage/' + Liste[i]
        #print(filename)
        if(Liste[i] != ".DS_Store"):
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
        if(Liste[i] != ".DS_Store"):
                Test_label[indice] = LabelNotation.index(Liste[i][0:4])
                indice = indice+1
                img = cv2.imread(filename,0)
                hist = (cv2.calcHist([img], [0], None, [256], [0, 256])).T
                Test[i,:]= (ComputeDist(Learn,hist)).T
        
    """
        TEST BASE TEST KPPV
    """
    K = 2
    Label_algo = np.zeros((len(Liste),K))
    Dist_algo = np.zeros((len(Liste),K))
    
    for i in range(0,len(Liste)):
        for j in range(0,K):
            Label_algo[i,j] = Label[np.argmin(Test[i,:])]
            Dist_algo[i,j] = Test[i,np.argmin(Test[i,:])]
            Test[i,np.argmin(Test[i,:])] = float('inf')
    
    return Label_algo,Test_label,Dist_algo

def main():
    

    #choix = int(input('resize image 1 oui / 2 non : '))
    choix = 10
    if(choix == 1):
        ResizeImage()
    else:
        print('No resize!')
    


    Liste = os.listdir('./BaseApprentissage')
    NombreImage = len(Liste) 

    Learn = []
    Label = []


    for i in range(0,NombreImage):
        filename = './BaseApprentissage/' + Liste[i]
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

    """
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
            #print(' label res premièer image : ' , LabelNotation[res[0]] , LabelNotation[res[1]] , LabelNotation[res[1]] , LabelNotation[res[1]] )
            if Liste[i][0:4] == LabelNotation[indiceMax]:
                sucess = sucess+1
    """
    Label_algo_histo, Test_label_histo, Dist_algo = Histoprocess()
    
    Liste = os.listdir('./BaseTest')
    NombreImage = len(Liste) 
    sucess = 0;
    nb_test =0;

    for i in range(0,NombreImage):
        filename = './BaseTest/' + Liste[i]
        if(Liste[i] !=".DS_Store"):
            nb_test =nb_test+1
            Features_test = Image_blocCompute_Predict(filename)

            print('Image : ',Liste[i])

            tab_Best_res = []
            for j in range(0,len(Features_test)):
                res = clf.predict(Features_test[j])
                indiceMax =0
                valeurIndiceMax =0



                for j in range(0,len(LabelNotation)):
                    #compteur classique
                    cpt = compteur(res,j)
                    #compteur qui jout avec l'importance des premiers points
                    #cpt = compteurPoids(res,j)
                    
                    if(cpt>valeurIndiceMax):
                        valeurIndiceMax=cpt
                        indiceMax =j

                
                
                tab_Best_res.append([indiceMax , valeurIndiceMax])
                print('               tour find : ' , LabelNotation[indiceMax] , ' ratio val trouve :' , valeurIndiceMax,'/',len(res))
        

            #traitement tableau des res
            tab_indice = []
            tab_score = []

            for j in range(0,len(tab_Best_res)):
                #rechcerche si l'indice na pas deja ete traite
                if(not tab_Best_res[j][0] in tab_indice):
                    tab_indice.append(tab_Best_res[j][0])
                    tab_score.append(tab_Best_res[j][1])
                else:
                    tab_score[tab_indice.index(tab_Best_res[j][0])] = tab_score[tab_indice.index(tab_Best_res[j][0])] + tab_Best_res[j][1]

            histo = Label_algo_histo[i]
            histo = [int(i) for i in histo]
            for k in range(0,len(histo)):
                print('               tour find histo : ' , LabelNotation[histo[k]])

            for j in range(0,len(tab_indice)):
                #parcour tableau label_algo_histo
                for k in range(0,len(histo)):
                    if(histo[k] == tab_indice[j]):
                        tab_score[j] = tab_score[j] + 10/(k+1)


            #recherche max dans tableau score pour determiner le label final
            val_max =0
            val_indiceMax =0
            for j in range(0,len(tab_score)):
                if(tab_score[j] > val_max):
                    val_max = tab_score[j]
                    val_indiceMax = tab_indice[j]
            print('               tour find : ' , LabelNotation[val_indiceMax])
            if Liste[i][0:4] == LabelNotation[val_indiceMax]:
                sucess = sucess+1
        

    

    return  sucess , nb_test

    


suc, res = main()
print(' success : ' , suc , ' / nb test : ' , res , ' / % : ' , (suc/res)*100)