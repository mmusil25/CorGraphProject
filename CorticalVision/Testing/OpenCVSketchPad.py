import numpy as np
import imutils
import cv2
import random
import itertools
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

import keras
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.datasets import cifar10
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


image = cv2.imread("TCS10.PNG")
image2 = cv2.imread("Ronaldo.jpg")
# print(image)
print(image.shape)
reshapedimage  = np.reshape(image, (3,312,312))
height,width = image2.shape[:2] 
resizedimage2 = cv2.resize(image2,(312,312))
reshapedimage2 = np.reshape(resizedimage2, (3,312,312))
#print(reshapedimage.shape)
#print(reshapedimage2.shape)

image3 = cv2.imread("Barcelona-star-Lionel-Messi-2.jpg")
resizedimage3 = cv2.resize(image3,(312,312))
reshapedimage3 = np.reshape(resizedimage3,(3,312,312))
arrayOfImages = np.stack((reshapedimage,reshapedimage2,reshapedimage3),axis=0)
#print(arrayOfImages.shape)
#print(arrayOfImages[0,:].shape)
reshapedimageout = np.reshape(arrayOfImages[1,:],(312,312,3))
#plt.imshow(reshapedimageout)
#plt.show()



(train_data,train_labels_one_hot),(test_data,test_labels_one_hot)=cifar10.load_data()

print(train_data.shape,train_labels_one_hot.shape)
print(train_labels_one_hot)

