from math import *
from PIL import Image
import numpy as np
import os

height = 1024

print('list des images')
Liste = os.listdir('../Images_dataset')


for i in range (1,len(Liste)):
    print(Liste[i])
    img = Image.open ('../Images_dataset/' +Liste[i])
    wpercent = (height / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    newImage = img.resize((height, hsize))
    print(newImage)
    newImage.save('./Image_data/'+ Liste[i])




