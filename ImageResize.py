from math import *
from PIL import Image
import numpy as np
import os

height = 1024
width = 768

print('list des images')
Liste = os.listdir('../Images_dataset')

print(Liste)

for i in range (1,len(Liste)):
    print(Liste[i])
    Im = Image.open ('../Images_dataset/' +Liste[i])
    print(Im);
    newImage = Im.resize((height,width))
    print(newImage)
    newImage.save('./Image_data/'+ Liste[i])




