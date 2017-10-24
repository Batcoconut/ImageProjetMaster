from math import *
from PIL import Image
import numpy as np
import os

height = 1024

<<<<<<< Updated upstream
print('list des images')
Liste = os.listdir('./BaseT')
=======
print('liste des images')
Liste = os.listdir('Image_data')
>>>>>>> Stashed changes


for i in range (1,len(Liste)):
    print(Liste[i])
<<<<<<< Updated upstream
    img = Image.open ('./BaseT/' +Liste[i])
=======
    img = Image.open ('Image_data/' +Liste[i])
>>>>>>> Stashed changes
    wpercent = (height / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    newImage = img.resize((height, hsize))
    print(newImage)
<<<<<<< Updated upstream
    newImage.save('./BaseTest/'+ Liste[i])
=======
    newImage.save('Image_data/'+ Liste[i])
>>>>>>> Stashed changes




