import numpy as np

def buildSDR()

	SquareIndex, CircleIndex = ShapeTypes[0],ShapeTypes[1]

	# Build the Triangle's base SDR. One hot encoding used for all SDRs.
	TriangleSDR = np.zeros(120) 
	# Part A: Number of Sides [0 - 19]
	TriangleSDR[2] = 1		
	# Part B: Height [20 - 39]

	# Part C: Width [40 - 59]
	
	# Part D: Number of Neighbors [60 - 79]
	TriangleSDR[42] = 1
	# Part E: Information of Neighbors [80 - 119]
		# Neighbor 1 - Circle [80 - 99] 
			# Part E.a: Number of Sides [80-86]
	TriangleSDR[81] = 1	
			# Part E.a: Height [87-94]

			# Part E.a: Width [95-99]
		# Neighbor 2 - Square [100 - 119]
			# Part E.a: Number of Sides [100 - 106]
	TriangleSDR[104] = 1
			# Part E.b: Height [107 - 114]

			# Part E.c: Width [114 - 119]

	# Build the Circle's base SDR

	CircleSDR = np.zeros(120) 
	# Part A: Number of Sides [0 - 19]
	CircleSDR[2] = 1		
	# Part B: Height [20 - 39]

	# Part C: Width [40 - 59]
					
	# Part D: Number of Neighbors [60 - 79]
	CircleSDR[42] = 1
	# Part E: Information of Neighbors [80 - 119]
		# Neighbor 1 - Triangle [80 - 99] 
			# Part E.a: Number of Sides [80-86]
	CircleSDR[83] = 1	
			# Part E.a: Height [87-94]

			# Part E.a: Width [95-99]
		# Neighbor 2 - Square [100 - 119]
			# Part E.a: Number of Sides [100 - 106]
	CircleSDR[104] = 1
			# Part E.b: Height [107 - 114]

			# Part E.c: Width [114 - 119]


	# Build the Square's base SDR

	SquareSDR = np.zeros(120) 
	# Part A: Number of Sides [0 - 19]
	SquareSDR[4] = 1		
	# Part B: Height [20 - 39]

	# Part C: Width [40 - 59]
					
	# Part D: Number of Neighbors [60 - 79]
	SquareSDR[42] = 1
	# Part E: Information of Neighbors [80 - 119]
		# Neighbor 1 - Triangle [80 - 99] 
			# Part E.a: Number of Sides [80-86]
	SquareSDR[83] = 1	
			# Part E.a: Height [87-94]

			# Part E.a: Width [95-99]
		# Neighbor 2 - Circle [100 - 119]
			# Part E.a: Number of Sides [100 - 106]
	SquareSDR[101] = 1
			# Part E.b: Height [107 - 114]
			# Part E.c: Width [114 - 119]



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
	

