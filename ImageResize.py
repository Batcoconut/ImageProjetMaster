from math import *
from PIL import Image
import numpy as np
import os

height = 1024



print('choix resize image Base Apprentissage : 1')
print('choix resize image Base test : 2')

test = True
choix = 0

folder = './BaseT'
folder_target = './BaseTest/'

while(test):
    choix = int(input('choix mode :  ' ))
    if choix == 1:
        break
    if choix == 2:
        break

if choix ==1:
    folder = './BaseAP'
    folder_target = './BaseApprentissage/'

print('list des images')
Liste = os.listdir(folder)

for i in range (1,len(Liste)):
    print(Liste[i])
    img = Image.open (folder + '/' +Liste[i])
    wpercent = (height / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    newImage = img.resize((height, hsize))
    print(newImage)
    newImage.save(folder_target + Liste[i])






