import numpy as np


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
	

