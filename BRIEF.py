#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:08:39 2017

@author: Marie Soret

compute the BRIEF descriptor of grayscale image
    INPUT :
        I : image to be describe, grayscale
        S : Size of the patch
        Nd : number of pair location

"""

from PIL import Image
import numpy as np
import scipy.ndimage

def BRIEF(I,S,Nd):
    
    # Size of the image
    [xlen,ylen]=I.size
    
    # Add black pad on the image's edges
    I_pad=Image.new('L', (xlen+2*S,ylen+2*S))
    I_pad.paste(I,(S,S))
    
    
    # Initialize descriptor
    B=np.zeros([xlen,ylen])
    
    # Treatment loop
    for x in range(S,S+xlen-1):
        for y in range(S,S+ylen-1):
            
            # Define patch of size SxS centered on pixel (x,y)
            patch=I_pad.crop((x-S, y-S, x+S, y+S))
            
            # Apply gaussian filter to smooth the image
            patch_smooth=scipy.ndimage.filters.gaussian_filter(patch, sigma=2.0)
            
            # Choose Nd random pair location pi et pj
            # GII : isitropic gaussian distribution
            mean=[S-1,S-1]
            var=S**2/25 
            cov=[[var,0],[0,var]]
            pi=np.random.multivariate_normal(mean,cov,Nd).astype(int)
            pj=np.random.multivariate_normal(mean,cov,Nd).astype(int)
            
            # Look for out of range indice
            a=np.where(pi<0)
            pi[a]=0
            b=np.where(pj<0)
            pj[b]=0
            c=np.where(pi>2*S-1)
            pi[c]=2*S-1
            d=np.where(pj>2*S-1)
            pj[d]=2*S-1
                        
            # Test 
            # Initialize test vector T
            T=np.zeros(Nd).astype(int)
            # Fill T
            for k in range (0,Nd-1):
                if patch_smooth[pi[k][0]][pi[k][1]]<patch_smooth[pj[k][0]][pj[k][1]]:
                    T[k]=1
                else:
                    T[k]=0
                # Update descripteur
                B[x-S,y-S]=B[x-S,y-S]+(2**k*T[k])
                
    B=np.reshape(B,[xlen*ylen,1])
            
    return B
