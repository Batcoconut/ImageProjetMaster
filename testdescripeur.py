# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from math import *
from PIL import Image
#import numpy as np
import os
import BRIEF
#import scipy.ndimage



# charger image en niveau de gris
Liste = os.listdir('Image_data')
I = Image.open('Image_data/'+Liste[1]).convert('L')
I_crop=I.crop((0,0,50,50))

# function parameterd
S=20
Nd=10

d=BRIEF.BRIEF(I_crop,S,Nd)  
                        
                                
                            
                
                

