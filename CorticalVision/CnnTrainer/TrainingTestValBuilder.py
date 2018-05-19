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



def buildSets(imageQuantity, inputSideLength):
		
	TrainingLimit, TestLimit, ValidationLimit = 400,450,500
	ListofImages = []

	for i in range(imageQuantity):
		image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Images/Triangle/T" + str(i) + ".PNG")
		if np.random.uniform(0,1) < 0.25:
			image =  cv2.blur(image,(5,5))
	#print(image)
	#print(image.shape)
	#plt.imshow(image)
	#plt.show()
		resizedImage = cv2.resize(image, (inputSideLength,inputSideLength))
		reshapedImage = np.reshape(resizedImage, (inputSideLength,inputSideLength,3))
		ListofImages.append(reshapedImage)
	TriangleImagesMaster = np.stack(ListofImages,axis=0)
# print(TriangleImagesMaster.shape)

#Triangle will be 1 in the CNN's final output

	TriangleKeyMaster = np.full(imageQuantity,0,dtype=int)


#  *===================*
#  * Circle Processing *
#  *===================*
	ListofImages = []

	for i in range(imageQuantity):
		image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Images/Circle/C" + str(i) + ".PNG")
		if np.random.uniform(0,1) < 0.25:
			image =  cv2.blur(image,(5,5))
	#print(image)
	#print(image.shape)
	#plt.imshow(image)
	#plt.show()
		resizedImage = cv2.resize(image, (inputSideLength,inputSideLength))
		reshapedImage = np.reshape(resizedImage, (inputSideLength,inputSideLength,3))
		ListofImages.append(reshapedImage)
	CircleImagesMaster = np.stack(ListofImages,axis=0)

#Circle will be 2 in the CNN's final output

	CircleKeyMaster = np.full(imageQuantity,1,dtype=int)

#  *===================*
#  * Square Processing *
#  *===================*

	ListofImages = []

	for i in range(imageQuantity):
		image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Images/Square/S" + str(i) + ".PNG")
		if np.random.uniform(0,1) < 0.25:
			image =  cv2.blur(image,(5,5))
	
	#print(image)
	#print(image.shape)
	#plt.imshow(image)
	#plt.show()
		resizedImage = cv2.resize(image, (inputSideLength,inputSideLength))
		reshapedImage = np.reshape(resizedImage, (inputSideLength,inputSideLength,3))
		ListofImages.append(reshapedImage)
	SquareImagesMaster = np.stack(ListofImages,axis=0)

#Square will be 3 in the CNN's final output

	SquareKeyMaster = np.full(imageQuantity,2,dtype=int)

# *===================*
# * Building the Sets *
# *===================*

# Slice the first 30 of each set to be for training

	TriangleImgsTrain = TriangleImagesMaster[:TrainingLimit]
# print(TriangleImgsTrain.shape)
	TriangleKeyTrain = TriangleKeyMaster[:TrainingLimit] 
#print(TriangleKeyTrain)

	CircleImgsTrain = CircleImagesMaster[:TrainingLimit]
	CircleKeyTrain = CircleKeyMaster[:TrainingLimit]

	SquareImgsTrain = SquareImagesMaster[:TrainingLimit]
	SquareKeyTrain = SquareKeyMaster[:TrainingLimit]

# Slice another 10 for testing

	TriangleImgsTest = TriangleImagesMaster[TrainingLimit:TestLimit]
#print(TriangleImgsTrain.shape)
	TriangleKeyTest = TriangleKeyMaster[TrainingLimit:TestLimit] 
#print(TriangleKeyTrain)

	CircleImgsTest = CircleImagesMaster[TrainingLimit:TestLimit]
	CircleKeyTest = CircleKeyMaster[TrainingLimit:TestLimit]

	SquareImgsTest = SquareImagesMaster[TrainingLimit:TestLimit]
	SquareKeyTest = SquareKeyMaster[TrainingLimit:TestLimit]

# And another 10 for validation 

	TriangleImgsValid = TriangleImagesMaster[TestLimit:ValidationLimit]
#print(TriangleImgsTrain.shape)
	TriangleKeyValid = TriangleKeyMaster[TestLimit:ValidationLimit] 
#print(TriangleKeyTrain)

	CircleImgsValid = CircleImagesMaster[TestLimit:ValidationLimit]
	CircleKeyValid = CircleKeyMaster[TestLimit:ValidationLimit]

	SquareImgsValid = SquareImagesMaster[TestLimit:ValidationLimit]
	SquareKeyValid = SquareKeyMaster[TestLimit:ValidationLimit]

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



