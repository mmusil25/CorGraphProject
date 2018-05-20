import numpy as np
import imutils
import cv2
import random
import itertools
import networkx as nx
import keras
import sys
import nupic
import os
from nupic.algorithms.spatial_pooler import SpatialPooler
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model

model = load_model('/home/mark/Documents/CorGraphProjectGit/CorticalVision/CnnTrainer/TrainedModel5')
shapeDict = {0:'Triangle',1:'Circle',2:'Square'}

# *=======================*
# *      Read image       *
# *=======================*
image = cv2.imread("TCS10.PNG")

# Create binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# *=========================*
# *      Find Contours      *
# *=========================*
(_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Found %d components." % len(contours))
centroids = []
G = nx.Graph()
tempInt = 0
listofImages = []
bufferRect = 0
bufferCNNInput = 30

# *==================================*
# *      Find Key SDR Properties     *
# *==================================*

for c in contours:
	listOfAreas = []	
	M = cv2.moments(c)
	temp = []
	temp.append(int(M["m10"] / M["m00"]))
	temp.append(int(M["m01"] / M["m00"]))
	x,y,w,h = cv2.boundingRect(c)

#======== Find the Area
	listOfAreas.append(M["m00"])
#======== Find Distance between the contours

#== Delta C0, C1

#== Delta C1, C2

#== Delta C2, C0
	
#======== Cut out the images and prepare them for the CNN

	cropped = image[y-bufferCNNInput:y+h+bufferCNNInput,x-bufferCNNInput:x+w+bufferCNNInput]
	plt.imshow(cropped)
	plt.show()
	resizedImage = cv2.resize(cropped,(32,32))
	reshapedImage = np.reshape(resizedImage,(32,32,3))
	listofImages.append(reshapedImage)
	cv2.rectangle(image,(x-bufferRect,y-bufferRect),(x+w+bufferRect,y+h+bufferRect),(0,255,0),2)

#======== Miscellaneous

	centroids.append(temp)
	# Adding nodes to the graph with their attributes
	G.add_node(tempInt, pos = temp)
	tempInt = tempInt + 1

#======= Pass the shapes to the CNN to find their shape types 

CNNInput = np.stack(listofImages,axis=0)
shapeTypes = model.predict(CNNInput)
listOfShapeTypes = []
for i in range(3):
	max = np.argmax(shapeTypes[i])
	print[shapeDict[max]]
	listOfShapeTypes.append(max)
	
#   *=========================*
#   *      Build The SDR      *
#   *=========================*



#   *=================================*
#   *      Push Through SP Layers     *
#   *=================================*

#=== SP Setup


#=== Layer 1


#=== Save the model












