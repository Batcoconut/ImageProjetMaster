import cv2
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import os

##ouverture image 


Image = [ 'noIsir','Isir','noIsir']

def ComputeFeatures(image_filename):
    img = cv2.imread(filename,0)
    ##Detection des surf points
    orb = cv2.ORB_create()
    pts , Features = orb.detectAndCompute(img,None)
    test = Features.flat[:]
    return test
    



Liste = os.listdir('./test')
Learn = np.zeros((3,16000))
Test = np.zeros((1,16000))


for i in range(0,len(Liste)):
    filename = './test/' + Liste[i]
    
    tab = ComputeFeatures(filename)
    if i == 3:
        print('test ok : ' + filename)
        Test[0] = tab
        print(tab)
    else:
        print('base apprentissage')
        Learn[i] = tab
    





print(Learn)
print(Test)
    


Label = np.array([0,1,2])
print(Label.shape)
print(Learn.shape)


clf = NearestCentroid()
clf.fit(Learn,Label)

#test predict
indice = clf.predict(Test)
print(Image[indice[0]])




