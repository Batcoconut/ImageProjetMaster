#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 22:50:56 2017

@author: alexis
"""

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

array = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,2,0,0,0,0,0,0,0,0,0,0],
[0,0,0,2,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,0,0,1,0,1,0],
[0,0,0,0,0,1,0,0,1,0,1,1,1],
[0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0],
[2,0,0,0,1,0,0,0,0,0,0,0,1],
[0,0,2,0,0,0,0,0,0,0,1,0,1],
[0,0,1,1,0,0,0,0,2,0,0,0,0],
[0,0,2,0,0,0,0,1,0,0,0,1,1]]

df_cm = pd.DataFrame(array, range(13),
                  range(13))
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size