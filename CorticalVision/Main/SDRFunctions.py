import numpy as np
import nupic
from nupic.algorithms.spatial_pooler import SpatialPooler
from nupic.encoders.scalar import ScalarEncoder


def buildSDR(arrayOfShapeTypes,arrayOfDistances,arrayOfAreas,numCentroids):

#Scalar Encoder for the area
	areaEncoder = ScalarEncoder(5, 1e3, 3e3, periodic=False, n=80, radius=0, resolution=0, name=None, verbosity=0, clipInput=False, forced=True)

#Scalar encoder for the distances between centroids

	distanceEncoder = ScalarEncoder(5, 30, 220, periodic=False, n=80, radius=0, resolution=0, name=None, verbosity=0, clipInput=False, forced=True)

	distanceBits0 = np.zeros(distanceEncoder.getWidth())
	distanceEncoder.encodeIntoArray(arrayOfDistances[0],distanceBits0)

	distanceBits1 = np.zeros(distanceEncoder.getWidth())
	distanceEncoder.encodeIntoArray(arrayOfDistances[1],distanceBits1)

	distanceBits2 = np.zeros(distanceEncoder.getWidth())
	distanceEncoder.encodeIntoArray(arrayOfDistances[2],distanceBits2)

# Build the Triangle's base SDR. One hot encoding used for all SDRs.
	TriangleSDR = np.zeros(10) 
	# Part A: Number of Sides [0 - 4]
	TriangleSDR[2] = 1		
	# Part B: Number of Neighbors [5 - 9]
	TriangleSDR[7] = 1

# Build the Circle's base SDR
	CircleSDR = np.zeros(10) 
	# Part A: Number of Sides [0 - 4]
	CircleSDR[0] = 1	
	# Part B: Number of Neighbors [5 - 9]
	CircleSDR[7] = 1

# Build the Square's base SDR
	SquareSDR = np.zeros(10) 
	# Part A: Number of Sides [0 - 4]
	SquareSDR[3] = 1	
	# Part B: Number of Neighbors [5 - 9]
	SquareSDR[7] = 1	



	arrayOfSDRs = np.array([])
	for i in range(numCentroids):

# Encode the area
		areaBits = np.zeros(areaEncoder.getWidth())
		areaEncoder.encodeIntoArray(arrayOfAreas[i],areaBits)
		
		# Figure out the shape type via the CNN output then us an		
		# if, elseif tree to decide which SDR to concatenate
		if arrayOfShapeTypes[i] == 0: #Its a Triangle
			tempSDR = np.concatenate(( TriangleSDR, areaBits))
		elif arrayOfShapeTypes[i] == 1: #Its a Circle
			tempSDR = np.concatenate(( CircleSDR, areaBits))
		elif arrayOfShapeTypes[i] == 2: #Its a Square
			tempSDR = np.concatenate(( SquareSDR, areaBits))

		if i == 0: #Its the first item
			tempSDR = np.concatenate((tempSDR,distanceBits0,distanceBits2))
		elif i == 1: #Its the second
			tempSDR = np.concatenate((tempSDR,distanceBits0,distanceBits1))
		elif i == 2: #Its the third
			tempSDR = np.concatenate((tempSDR,distanceBits1,distanceBits2))
	#	print(tempSDR)
		arrayOfSDRs = np.append(arrayOfSDRs,tempSDR)
		#print(tempSDR.shape)
		
		if numCentroids == 2:
			arrayOfSDRs = np.append(arrayOfSDRs,np.zeros(250))
		if numCentroids == 1:
			arrayOfSDRs = np.append(arrayOfSDRs,np.zeros(500))
# Concatenate all three SDRs
	#print(arrayOfSDRs)
	imageSDR = arrayOfSDRs

	return imageSDR

def add(self, other):
    self[0] = self[0] + other[0]
    self[1] = self[1] + other[1]

def divide(self, number):
	return(self[0] / number, self[1] / number)

# To draw graph
def drawGraph(self, nColor, eColor):
	pos = nx.get_node_attributes(self,'pos')
	nx.draw_networkx(self, pos, node_color = nColor, edge_color = eColor)

# To construct graph edges
def Graph_EdgesConstruction(centers, Graph, radius):

	# Applying KNN for computing neighbors
	neigh = NearestNeighbors(radius=radius).fit(centers)
	distances, indices = neigh.radius_neighbors(centers)

	# Creating edges with the help of neighbor's indices
	a_list = []
	for ind in indices:
		for subset in itertools.combinations(ind, 2):
			a_list.append(tuple(subset))
	a = list(set(a_list))
	Graph.add_edges_from(a)
	return Graph

def DetectShape(ListIn, NumComponents):
	# Find the shape with minimal contours, this will be the square.
	minimum = len(ListIn[1])
	for i in range(NumComponents):
		if ListIn[i].size < minimum:
			minimum = ListIn[i].size
			SquareIndex = i
	# Find the average slope error for the other two
	AvgSlopeErrors = np.array([])
	KList = np.array([]) 	 
	for k in range(NumComponents):
		ListOfSlopeErrorsPerShape = np.array([])
		if k != SquareIndex:
			for l in range(len(ListIn[k])-2):
				# print(ListIn[k][l+1])
				# print(ListIn[k][l+1][0][0])
				# print(ListIn[k][l+1][0][1])	
				
				slope1 = (ListIn[k][l+1][0][1]-ListIn[k][l][0][1])/(ListIn[k][l+1][0][0]-ListIn[k][l][0][0]+0.04)
				slope2 = (ListIn[k][l+2][0][1]-ListIn[k][l+1][0][1])/(ListIn[k][l+2][0][0]-ListIn[k][l+1][0][0]+0.04)
				ListOfSlopeErrorsPerShape = np.append(ListOfSlopeErrorsPerShape, np.absolute(slope2-slope1))	
			AvgSlopeError = np.sum(ListOfSlopeErrorsPerShape)/ListOfSlopeErrorsPerShape.size
		#	print(AvgSlopeError)
			AvgSlopeErrors = np.append(AvgSlopeErrors, AvgSlopeError)
			KList = np.array([KList, k])
		#	print(AvgSlopeErrors)
	LeastSlopeError = np.argmin(AvgSlopeErrors)
	# print(KList[LeastSlopeError])
	TriangleIndex = KList[LeastSlopeError]
	# Determine the Circle's Index by Proces of Elimination 
	# Need to modify this code for the non-three type case
	if ((SquareIndex == 0) & (TriangleIndex == 1)):
		CircleIndex = 2
	elif ((SquareIndex == 0) & (TriangleIndex == 2)):
		CircleIndex =  1
	elif ((SquareIndex == 1) & (TriangleIndex == 2)):
		CircleIndex = 0  
	
	Output = np.array([TriangleIndex, SquareIndex, CircleIndex])
	return Output
			
		
		
		
		# SlopeErrors = np.append([AvgSlopeError,k],axis=1)
	 	# MinSlopeError = np.ndarray.argmin(SlopeErrors, axis=0)	
			
		
		# ListOfLengths = numpy.append(ListOfLengths,len(ListIn[i]))
		# min = numpy.argmin(ListOfLengths)

#==========================================
def DetectShape(ListIn, NumComponents):
	# Find the shape with minimal contours, this will be the square.
	minimum = len(ListIn[1])
	SquareIndex = 0
	for i in range(NumComponents):
		if ListIn[i].size < minimum:
			minimum = ListIn[i].size
			SquareIndex = i
	# Find the average slope error for the other two
	AvgSlopeErrors = np.array([])
	KList = np.array([]) 	 
	for k in range(NumComponents):
		ListOfSlopeErrorsPerShape = np.array([])
		if k != SquareIndex:
			for l in range(len(ListIn[k])-2):
				# print(ListIn[k][l+1])
				# print(ListIn[k][l+1][0][0])
				# print(ListIn[k][l+1][0][1])	
				
				slope1 = (ListIn[k][l+1][0][1]-ListIn[k][l][0][1])/(ListIn[k][l+1][0][0]-ListIn[k][l][0][0]+0.04)
				slope2 = (ListIn[k][l+2][0][1]-ListIn[k][l+1][0][1])/(ListIn[k][l+2][0][0]-ListIn[k][l+1][0][0]+0.04)
				ListOfSlopeErrorsPerShape = np.append(ListOfSlopeErrorsPerShape, np.absolute(slope2-slope1))	
			AvgSlopeError = np.sum(ListOfSlopeErrorsPerShape)/ListOfSlopeErrorsPerShape.size
		#	print(AvgSlopeError)
			AvgSlopeErrors = np.append(AvgSlopeErrors, AvgSlopeError)
			KList = np.append(KList, k)
		#	print(AvgSlopeErrors)
	print(AvgSlopeErrors)
	LeastSlopeError = np.argmin(AvgSlopeErrors)
	print(LeastSlopeError)
	# print(KList[LeastSlopeError])
	TriangleIndex = int(KList[LeastSlopeError])
	print(TriangleIndex)
	print(SquareIndex)
	# Determine the Circle's Index by Proces of Elimination 
	# Need to modify this code for the non-three type case
	if ((SquareIndex == 0) & (TriangleIndex == 1)):
		CircleIndex = 2
	elif ((SquareIndex == 0) & (TriangleIndex == 2)):
		CircleIndex =  1
	elif ((SquareIndex == 1) & (TriangleIndex == 2)):
		CircleIndex = 0  
	elif ((SquareIndex == 1) & (TriangleIndex ==0)):
		CircleIndex = 2
	elif ((SquareIndex == 2) & (TriangleIndex == 0)):
		CircleIndex =  1
	elif ((SquareIndex == 2) & (TriangleIndex == 1)):
		CircleIndex = 0  


	Output = np.array([TriangleIndex, SquareIndex, CircleIndex])
	return Output
#===========================================================================			
def DetectShapeCS(ListIn, NumComponents):
	# Find the shape with minimal contours, this will be the square.
	minimum = len(ListIn[1])
	SquareIndex = 0
	for i in range(NumComponents):
		if ListIn[i].size < minimum:
			minimum = ListIn[i].size
			SquareIndex = i

# Circle index is just the other index
	if (SquareIndex == 0):
		CircleIndex = 1
	elif (SquareIndex == 1):
		CircleIndex =  0

	Output = np.array([SquareIndex, CircleIndex])
	return Output
#=======================================================================
def DetectShapeTC(ListIn, NumComponents):
	# Find the shape with minimal contours, this will be the square.
	minimum = len(ListIn[1])
	SquareIndex = 0
	for i in range(NumComponents):
		if ListIn[i].size < minimum:
			minimum = ListIn[i].size
			SquareIndex = i
	# Find the average slope error for the other two
	AvgSlopeErrors = np.array([])
	KList = np.array([]) 	 
	for k in range(NumComponents):
		ListOfSlopeErrorsPerShape = np.array([])
		if k != SquareIndex:
			for l in range(len(ListIn[k])-2):
				# print(ListIn[k][l+1])
				# print(ListIn[k][l+1][0][0])
				# print(ListIn[k][l+1][0][1])	
				
				slope1 = (ListIn[k][l+1][0][1]-ListIn[k][l][0][1])/(ListIn[k][l+1][0][0]-ListIn[k][l][0][0]+0.04)
				slope2 = (ListIn[k][l+2][0][1]-ListIn[k][l+1][0][1])/(ListIn[k][l+2][0][0]-ListIn[k][l+1][0][0]+0.04)
				ListOfSlopeErrorsPerShape = np.append(ListOfSlopeErrorsPerShape, np.absolute(slope2-slope1))	
			AvgSlopeError = np.sum(ListOfSlopeErrorsPerShape)/ListOfSlopeErrorsPerShape.size
		#	print(AvgSlopeError)
			AvgSlopeErrors = np.append(AvgSlopeErrors, AvgSlopeError)
			KList = np.append(KList, k)
		#	print(AvgSlopeErrors)
	print(AvgSlopeErrors)
	LeastSlopeError = np.argmin(AvgSlopeErrors)
	print(LeastSlopeError)
	# print(KList[LeastSlopeError])
	TriangleIndex = int(KList[LeastSlopeError])
	print(TriangleIndex)
	print(SquareIndex)
	# Determine the Circle's Index by Proces of Elimination 
	# Need to modify this code for the non-three type case
	if ((SquareIndex == 0) & (TriangleIndex == 1)):
		CircleIndex = 2
	elif ((SquareIndex == 0) & (TriangleIndex == 2)):
		CircleIndex =  1
	elif ((SquareIndex == 1) & (TriangleIndex == 2)):
		CircleIndex = 0  
	elif ((SquareIndex == 1) & (TriangleIndex ==0)):
		CircleIndex = 2
	elif ((SquareIndex == 2) & (TriangleIndex == 0)):
		CircleIndex =  1
	elif ((SquareIndex == 2) & (TriangleIndex == 1)):
		CircleIndex = 0  


	Output = np.array([TriangleIndex, SquareIndex, CircleIndex])
	return Output
#=======================================================================
def DetectShapeTS(ListIn, NumComponents):
	# Find the shape with minimal contours, this will be the square.
	minimum = len(ListIn[1])
	SquareIndex = 0
	for i in range(NumComponents):
		if ListIn[i].size < minimum:
			minimum = ListIn[i].size
			SquareIndex = i
	# Find the average slope error for the other two
	AvgSlopeErrors = np.array([])
	KList = np.array([]) 	 
	for k in range(NumComponents):
		ListOfSlopeErrorsPerShape = np.array([])
		if k != SquareIndex:
			for l in range(len(ListIn[k])-2):
				# print(ListIn[k][l+1])
				# print(ListIn[k][l+1][0][0])
				# print(ListIn[k][l+1][0][1])	
				
				slope1 = (ListIn[k][l+1][0][1]-ListIn[k][l][0][1])/(ListIn[k][l+1][0][0]-ListIn[k][l][0][0]+0.04)
				slope2 = (ListIn[k][l+2][0][1]-ListIn[k][l+1][0][1])/(ListIn[k][l+2][0][0]-ListIn[k][l+1][0][0]+0.04)
				ListOfSlopeErrorsPerShape = np.append(ListOfSlopeErrorsPerShape, np.absolute(slope2-slope1))	
			AvgSlopeError = np.sum(ListOfSlopeErrorsPerShape)/ListOfSlopeErrorsPerShape.size
		#	print(AvgSlopeError)
			AvgSlopeErrors = np.append(AvgSlopeErrors, AvgSlopeError)
			KList = np.append(KList, k)
		#	print(AvgSlopeErrors)
	print(AvgSlopeErrors)
	LeastSlopeError = np.argmin(AvgSlopeErrors)
	print(LeastSlopeError)
	# print(KList[LeastSlopeError])
	TriangleIndex = int(KList[LeastSlopeError])
	print(TriangleIndex)
	print(SquareIndex)
	# Determine the Circle's Index by Proces of Elimination 
	# Need to modify this code for the non-three type case
	if ((SquareIndex == 0) & (TriangleIndex == 1)):
		CircleIndex = 2
	elif ((SquareIndex == 0) & (TriangleIndex == 2)):
		CircleIndex =  1
	elif ((SquareIndex == 1) & (TriangleIndex == 2)):
		CircleIndex = 0  
	elif ((SquareIndex == 1) & (TriangleIndex ==0)):
		CircleIndex = 2
	elif ((SquareIndex == 2) & (TriangleIndex == 0)):
		CircleIndex =  1
	elif ((SquareIndex == 2) & (TriangleIndex == 1)):
		CircleIndex = 0  


	Output = np.array([TriangleIndex, SquareIndex, CircleIndex])
	return Output
		
		
		
		# SlopeErrors = np.append([AvgSlopeError,k],axis=1)
	 	# MinSlopeError = np.ndarray.argmin(SlopeErrors, axis=0)	
			
		
		# ListOfLengths = numpy.append(ListOfLengths,len(ListIn[i]))
		# min = numpy.argmin(ListOfLengths)
	

