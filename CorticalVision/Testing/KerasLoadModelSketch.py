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
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
resizedimage = cv2.resize(image,(312,312))
reshapedimage2 = np.reshape(resizedimage2, (3,312,312))


(train_data,train_labels_one_hot),(test_data,test_labels_one_hot)=cifar10.load_data()

#print(train_data.shape,train_labels_one_hot.shape)
#print(train_labels_one_hot)

# Load pretrained keras model and try it out on a few images







