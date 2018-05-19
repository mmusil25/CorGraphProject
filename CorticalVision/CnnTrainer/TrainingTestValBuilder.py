# File for processing the raw images and turning them into training, testing, and validation sets

import numpy as np
import imutils
import cv2
import random
import itertools
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors

#  *=====================*
#  * Triangle Processing *
#  *=====================*

def buildSets():


	ListofImages = []

	for i in range(50):
		image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Images/Triangle/T" + str(i) + ".PNG")
	#print(image)
	#print(image.shape)
	#plt.imshow(image)
	#plt.show()
		resizedImage = cv2.resize(image, (78,78))
		reshapedImage = np.reshape(resizedImage, (3,78,78))
		ListofImages.append(reshapedImage)
	TriangleImagesMaster = np.stack(ListofImages,axis=0)
# print(TriangleImagesMaster.shape)

#Triangle will be 1 in the CNN's final output

	TriangleKeyMaster = np.full(50,1,dtype=int)


#  *===================*
#  * Circle Processing *
#  *===================*
	ListofImages = []

	for i in range(50):
		image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Images/Circle/C" + str(i) + ".PNG")
	#print(image)
	#print(image.shape)
	#plt.imshow(image)
	#plt.show()
		resizedImage = cv2.resize(image, (78,78))
		reshapedImage = np.reshape(resizedImage, (3,78,78))
		ListofImages.append(reshapedImage)
	CircleImagesMaster = np.stack(ListofImages,axis=0)

#Circle will be 2 in the CNN's final output

	CircleKeyMaster = np.full(50,2,dtype=int)

#  *===================*
#  * Square Processing *
#  *===================*

	ListofImages = []

	for i in range(50):
		image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Images/Square/S" + str(i) + ".PNG")
	#print(image)
	#print(image.shape)
	#plt.imshow(image)
	#plt.show()
		resizedImage = cv2.resize(image, (78,78))
		reshapedImage = np.reshape(resizedImage, (3,78,78))
		ListofImages.append(reshapedImage)
	SquareImagesMaster = np.stack(ListofImages,axis=0)

#Square will be 3 in the CNN's final output

	SquareKeyMaster = np.full(50,3,dtype=int)

# *===================*
# * Building the Sets *
# *===================*

# Slice the first 30 of each set to be for training

	TriangleImgsTrain = TriangleImagesMaster[:30]
# print(TriangleImgsTrain.shape)
	TriangleKeyTrain = TriangleKeyMaster[:30] 
#print(TriangleKeyTrain)

	CircleImgsTrain = CircleImagesMaster[:30]
	CircleKeyTrain = CircleKeyMaster[:30]

	SquareImgsTrain = SquareImagesMaster[:30]
	SquareKeyTrain = SquareKeyMaster[:30]

# Slice another 10 for testing

	TriangleImgsTest = TriangleImagesMaster[30:40]
#print(TriangleImgsTrain.shape)
	TriangleKeyTest = TriangleKeyMaster[30:40] 
#print(TriangleKeyTrain)

	CircleImgsTest = CircleImagesMaster[30:40]
	CircleKeyTest = CircleKeyMaster[30:40]

	SquareImgsTest = SquareImagesMaster[30:40]
	SquareKeyTest = SquareKeyMaster[30:40]

# And another 10 for validation 

	TriangleImgsValid = TriangleImagesMaster[40:50]
#print(TriangleImgsTrain.shape)
	TriangleKeyValid = TriangleKeyMaster[40:50] 
#print(TriangleKeyTrain)

	CircleImgsValid = CircleImagesMaster[40:50]
	CircleKeyValid = CircleKeyMaster[40:50]

	SquareImgsValid = SquareImagesMaster[40:50]
	SquareKeyValid = SquareKeyMaster[40:50]

# Build the final sets

# Training =======================

	TrainingImages = np.concatenate((TriangleImgsTrain,CircleImgsTrain,SquareImgsTrain),axis=0)
	TrainingKey = np.concatenate((TriangleKeyTrain,CircleKeyTrain,SquareKeyTrain),axis=0)
	#TrainingSet = np.stack((TrainingImages,TrainingKey),axis=0)
	#print(TrainingSet.shape)
# Testing =======================

	TestingImages = np.concatenate((TriangleImgsTest,CircleImgsTest,SquareImgsTest),axis=0)
	TestingKey = np.concatenate((TriangleKeyTest,CircleKeyTest,SquareKeyTest),axis=0)
	#TestingSet = np.stack((TestingImages,TestingKey),axis=0)

# Validation ===================

	ValidationImages = np.concatenate((TriangleImgsValid,CircleImgsValid,SquareImgsValid),axis=0)
	ValidationKey = np.concatenate((TriangleKeyValid,CircleKeyValid,SquareKeyValid),axis=0)
	#ValidationSet = np.stack((ValidationImages,ValidationKey),axis=0)
	
	return (TrainingImages, TrainingKey), (TestingImages, TestingKey), (ValidationImages, ValidationKey)



