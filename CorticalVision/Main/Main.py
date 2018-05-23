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
import SDRFunctions
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.encoders.scalar import ScalarEncoder
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model

model = load_model('/home/mark/Documents/CorGraphProjectGit/CorticalVision/CnnTrainer/TrainedModel5')
shapeDict = {0:'Triangle',1:'Circle',2:'Square'}
numInputs = 20
numShapes = 3
arrayOfSDRs = np.array([])
tempArrayOfSDRs = []

for i in range(numInputs):
# *=======================*
# *      Read image       *
# *=======================*
	image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Main/TriangleSquareCircle/TCS" + str(i) + ".PNG")
	#image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Main/TriangleSquareCircle/TCS10.PNG")
	# Create binary image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# *=========================*
# *      Find Contours      *
# *=========================*
	(_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	centroids = []
	G = nx.Graph()
	tempInt = 0
	listofImages = []
	listOfAreas = []
	bufferRect = 0
	bufferCNNInput = 0

# *==================================*
# *      Find Key SDR Properties     *
# *==================================*


	for c in contours:
	
		M = cv2.moments(c)
		temp = []
		temp.append(int(M["m10"] / M["m00"]))
		temp.append(int(M["m01"] / M["m00"]))
		x,y,w,h = cv2.boundingRect(c)

#======== Find the Area
		listOfAreas.append(M["m00"])
		print('Area of the object:', M["m00"])
	
#======== Cut out the images and prepare them for the CNN

		cropped = image[y-bufferCNNInput:y+h+bufferCNNInput,x-bufferCNNInput:x+w+bufferCNNInput]
		#plt.imshow(cropped)
		#plt.show()
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

	arrayOfAreas = np.asarray(listOfAreas)
	CNNInput = np.stack(listofImages,axis=0)
	shapeTypes = model.predict(CNNInput)
	listOfShapeTypes = []

	for i in range(numShapes):
		max = np.argmax(shapeTypes[i])
		#print[shapeDict[max],listOfAreas[i]]
		listOfShapeTypes.append(max)
		print['Object ' + str(i) + ' is a',shapeDict[max]]
	arrayOfShapeTypes = np.asarray(listOfShapeTypes)

#======== Find Distance between the contours

	#== Delta C0, C1
	deltax,deltay = abs(centroids[0][0] - centroids[1][0]), abs(centroids[0][1] - centroids[1][1])
	delta01 = (deltax**2 + deltay**2)**0.5	
	print('Distance from center 0 to 1: ', delta01)

	#== Delta C1, C2
	deltax,deltay = abs(centroids[2][0] - centroids[1][0]), abs(centroids[2][1] - centroids[1][1])
	delta12 = (deltax**2 + deltay**2)**0.5
	print('Distance from center 1 to 2: ', delta12)

	#== Delta C2, C0
	deltax,deltay = abs(centroids[0][0] - centroids[2][0]), abs(centroids[0][1] - centroids[2][1])
	delta20 = (deltax**2 + deltay**2)**0.5
	print('Distance from center 0 to 2: ', delta20)

	arrayOfDistances = np.array([delta01,delta12,delta20]) 

#   *=========================*
#   *      Build The SDR      *
#   *=========================*

	imageSDR = SDRFunctions.buildSDR	(arrayOfShapeTypes,arrayOfDistances,arrayOfAreas,3)
#print(imageSDR)
#print(imageSDR.shape)
#print(np.count_nonzero(imageSDR))
	tempArrayOfSDRs.append(imageSDR)
arrayOfSDRs = np.stack(tempArrayOfSDRs,axis=0)
print(arrayOfSDRs.shape)
#   *=================================*
#   *      Push Through SP Layers     *
#   *=================================*

#=== Layer 1 SP Setup

EncodingWidth, SpatialPoolerWidth1 = 750, 600

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth1),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=30.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 1
activeColumns = np.zeros(SpatialPoolerWidth1)

#Push all image SDRs through the SP layer 1
SPLayer1Out = np.array([])
tempList = []
#print('L1 TriSqrCrcl Out: ')
for i in range(numInputs):
	sp.compute(arrayOfSDRs[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	#print(activeColumnIndices[0])
SPLayer1Out = np.stack(tempList,axis = 0)	
	

#=== Layer 2 SP Setup

EncodingWidth, SpatialPoolerWidth2 = SpatialPoolerWidth1, 400

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth2),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=20.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 2
activeColumns = np.zeros(SpatialPoolerWidth2)

#Push all image SDRs through the SP layer 2
SPLayer2Out = np.array([])
tempList = []
#print('L2 TriSqrCrcl Out: ')
for i in range(numInputs):
	sp.compute(SPLayer1Out[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	#print(activeColumnIndices[0])
SPLayer2Out = np.stack(tempList,axis = 0)

#=== Layer 2 SP Setup

EncodingWidth, SpatialPoolerWidth3 = SpatialPoolerWidth2, 200

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth3),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=10.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 2
activeColumns = np.zeros(SpatialPoolerWidth3)

#Push all image SDRs through the SP layer 2
SPLayer3Out = np.array([])
tempList = []
print('Layer 3 TriangleSquareCircle Out: ')
for i in range(numInputs):
	sp.compute(SPLayer2Out[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	print(activeColumnIndices[0])
SPLayer3Out = np.stack(tempList,axis = 0)





# *===================================================================*
# *      Begin Double Shape  Experiment                               *
# *===================================================================*

numInputs = 20
numShapes = 2
arrayOfSDRs = np.array([])
tempArrayOfSDRs = []

for i in range(numInputs):
# *=======================*
# *      Read image       *
# *=======================*
	image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Main/CircleSquare/CS" + str(i) + ".PNG")

	# Create binary image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# *=========================*
# *      Find Contours      *
# *=========================*
	(_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	centroids = []
	G = nx.Graph()
	tempInt = 0
	listofImages = []
	listOfAreas = []
	bufferRect = 0
	bufferCNNInput = 0

# *==================================*
# *      Find Key SDR Properties     *
# *==================================*


	for c in contours:
	
		M = cv2.moments(c)
		temp = []
		temp.append(int(M["m10"] / M["m00"]))
		temp.append(int(M["m01"] / M["m00"]))
		x,y,w,h = cv2.boundingRect(c)

#======== Find the Area
		listOfAreas.append(M["m00"])
	
#======== Cut out the images and prepare them for the CNN

		cropped = image[y-bufferCNNInput:y+h+bufferCNNInput,x-bufferCNNInput:x+w+bufferCNNInput]
		#plt.imshow(cropped)
		#plt.show()
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

	arrayOfAreas = np.asarray(listOfAreas)
	CNNInput = np.stack(listofImages,axis=0)
	shapeTypes = model.predict(CNNInput)
	listOfShapeTypes = []

	for i in range(numShapes):
		max = np.argmax(shapeTypes[i])
		#print[shapeDict[max],listOfAreas[i]]
		listOfShapeTypes.append(max)
	arrayOfShapeTypes = np.asarray(listOfShapeTypes)

#======== Find Distance between the contours

	#== Delta C0, C1
	deltax,deltay = abs(centroids[0][0] - centroids[1][0]), abs(centroids[0][1] - centroids[1][1])
	delta01 = (deltax**2 + deltay**2)**0.5

	#== Delta C1, C2
#	deltax,deltay = abs(centroids[2][0] - centroids[1][0]), abs(centroids[2][1] - centroids[1][1])
#	delta12 = (deltax**2 + deltay**2)**0.5

	#== Delta C2, C0
#	deltax,deltay = abs(centroids[0][0] - centroids[2][0]), abs(centroids[0][1] - centroids[2][1])

#	delta20 = (deltax**2 + deltay**2)**0.5

	arrayOfDistances = np.array([delta01]) 

#   *=========================*
#   *      Build The SDR      *
#   *=========================*

	imageSDR = SDRFunctions.buildSDR	(arrayOfShapeTypes,arrayOfDistances,arrayOfAreas,numShapes)
#print(imageSDR)
#print(imageSDR.shape)
#print(np.count_nonzero(imageSDR))
	tempArrayOfSDRs.append(imageSDR)
arrayOfSDRs = np.stack(tempArrayOfSDRs,axis=0)
print(arrayOfSDRs.shape)
#   *=================================*
#   *      Push Through SP Layers     *
#   *=================================*

#=== Layer 1 SP Setup

EncodingWidth, SpatialPoolerWidth1 = 750, 600

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth1),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=30.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 1
activeColumns = np.zeros(SpatialPoolerWidth1)

#Push all image SDRs through the SP layer 1
SPLayer1Out = np.array([])
tempList = []
#print('L1 CrclSqr Out: ')
for i in range(numInputs):
	sp.compute(arrayOfSDRs[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	#print(activeColumnIndices[0])
SPLayer1Out = np.stack(tempList,axis = 0)	
	

#=== Layer 2 SP Setup

EncodingWidth, SpatialPoolerWidth2 = SpatialPoolerWidth1, 400

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth2),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=20.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 2
activeColumns = np.zeros(SpatialPoolerWidth2)

#Push all image SDRs through the SP layer 2
SPLayer2Out = np.array([])
tempList = []
#print('L2 CrclSqr Out: ')
for i in range(numInputs):
	sp.compute(SPLayer1Out[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	#print(activeColumnIndices[0])
SPLayer2Out = np.stack(tempList,axis = 0)

#=== Layer 2 SP Setup

EncodingWidth, SpatialPoolerWidth3 = SpatialPoolerWidth2, 200

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth3),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=10.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 2
activeColumns = np.zeros(SpatialPoolerWidth3)

#Push all image SDRs through the SP layer 2
SPLayer3Out = np.array([])
tempList = []
print('Layer 3 CircleSquare Out: ')
for i in range(numInputs):
	sp.compute(SPLayer2Out[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	print(activeColumnIndices[0])
SPLayer3Out = np.stack(tempList,axis = 0)

# *===================================================================*
# *      Begin Single Shape  Experiment                               *
# *===================================================================*

numInputs = 20
numShapes = 1
arrayOfSDRs = np.array([])
tempArrayOfSDRs = []

for i in range(numInputs):
# *=======================*
# *      Read image       *
# *=======================*
	image = cv2.imread("/home/mark/Documents/CorGraphProjectGit/CorticalVision/Main/Circle/C" + str(i) + ".PNG")

	# Create binary image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# *=========================*
# *      Find Contours      *
# *=========================*
	(_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	centroids = []
	G = nx.Graph()
	tempInt = 0
	listofImages = []
	listOfAreas = []
	bufferRect = 0
	bufferCNNInput = 0

# *==================================*
# *      Find Key SDR Properties     *
# *==================================*


	for c in contours:
	
		M = cv2.moments(c)
		temp = []
		temp.append(int(M["m10"] / M["m00"]))
		temp.append(int(M["m01"] / M["m00"]))
		x,y,w,h = cv2.boundingRect(c)

#======== Find the Area
		listOfAreas.append(M["m00"])
	
#======== Cut out the images and prepare them for the CNN

		cropped = image[y-bufferCNNInput:y+h+bufferCNNInput,x-bufferCNNInput:x+w+bufferCNNInput]
		#plt.imshow(cropped)
		#plt.show()
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

	arrayOfAreas = np.asarray(listOfAreas)
	CNNInput = np.stack(listofImages,axis=0)
	shapeTypes = model.predict(CNNInput)
	listOfShapeTypes = []

	for i in range(numShapes):
		max = np.argmax(shapeTypes[i])
		#print[shapeDict[max],listOfAreas[i]]
		listOfShapeTypes.append(max)
	arrayOfShapeTypes = np.asarray(listOfShapeTypes)

#======== Find Distance between the contours

	#== Delta C0, C1
#	deltax,deltay = abs(centroids[0][0] - centroids[1][0]), abs(centroids[0][1] - centroids[1][1])
#	delta01 = (deltax**2 + deltay**2)**0.5

	#== Delta C1, C2
#	deltax,deltay = abs(centroids[2][0] - centroids[1][0]), abs(centroids[2][1] - centroids[1][1])
#	delta12 = (deltax**2 + deltay**2)**0.5

	#== Delta C2, C0
#	deltax,deltay = abs(centroids[0][0] - centroids[2][0]), abs(centroids[0][1] - centroids[2][1])
#	delta20 = (deltax**2 + deltay**2)**0.5

	arrayOfDistances = np.array([delta01]) 

#   *=========================*
#   *      Build The SDR      *
#   *=========================*

	imageSDR = SDRFunctions.buildSDR	(arrayOfShapeTypes,arrayOfDistances,arrayOfAreas,numShapes)
#print(imageSDR)
#print(imageSDR.shape)
#print(np.count_nonzero(imageSDR))
	tempArrayOfSDRs.append(imageSDR)
arrayOfSDRs = np.stack(tempArrayOfSDRs,axis=0)
print(arrayOfSDRs.shape)
#   *=================================*
#   *      Push Through SP Layers     *
#   *=================================*

#=== Layer 1 SP Setup

EncodingWidth, SpatialPoolerWidth1 = 750, 600

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth1),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=30.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 1
activeColumns = np.zeros(SpatialPoolerWidth1)

#Push all image SDRs through the SP layer 1
SPLayer1Out = np.array([])
tempList = []
#print('L1 Crcl Out: ')
for i in range(numInputs):
	sp.compute(arrayOfSDRs[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	#print(activeColumnIndices[0])
SPLayer1Out = np.stack(tempList,axis = 0)	
	

#=== Layer 2 SP Setup

EncodingWidth, SpatialPoolerWidth2 = SpatialPoolerWidth1, 400

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth2),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=20.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 2
activeColumns = np.zeros(SpatialPoolerWidth2)

#Push all image SDRs through the SP layer 2
SPLayer2Out = np.array([])
tempList = []
#print('L2 Crcl Out: ')
for i in range(numInputs):
	sp.compute(SPLayer1Out[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	#print(activeColumnIndices[0])
SPLayer2Out = np.stack(tempList,axis = 0)

#=== Layer 2 SP Setup

EncodingWidth, SpatialPoolerWidth3 = SpatialPoolerWidth2, 200

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth3),
  # What percent of the columns's receptive field is available for potential
  # synapses?
  potentialPct=0.85,
  # This means that the input space has no topology.
  globalInhibition=True,
  localAreaDensity=-1.0,
  # Roughly 2%, giving that there is only one inhibition area because we have
  # turned on globalInhibition (40 / 2048 = 0.0195)
  numActiveColumnsPerInhArea=10.0,
  # How quickly synapses grow and degrade.
  synPermInactiveDec=0.005,
  synPermActiveInc=0.04,
  synPermConnected=0.1,
  # boostStrength controls the strength of boosting. Boosting encourages
  # efficient usage of SP columns.
  boostStrength=3.0,
  # Random number generator seed.
  seed=1956,
  # Determines if inputs at the beginning and end of an input dimension should
  # be considered neighbors when mapping columns to inputs.
  wrapAround=False
)

# Array which contains the output of the spatial pooler for layer 2
activeColumns = np.zeros(SpatialPoolerWidth3)

#Push all image SDRs through the SP layer 2
SPLayer3Out = np.array([])
tempList = []
print('Layer 3 Circle Out: ')
for i in range(numInputs):
	sp.compute(SPLayer2Out[i,:],True,activeColumns)
	tempList.append(activeColumns)
	#Prepare the output for printing
	activeColumnIndices = np.nonzero(activeColumns)
	print(activeColumnIndices[0])
SPLayer3Out = np.stack(tempList,axis = 0)














