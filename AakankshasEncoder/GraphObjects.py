import requests
import numpy as np
import imutils
import cv2
import random
import itertools
import networkx as nx
import SDRFunctions
import csv 
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors


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



for m in range(20):
	image = cv2.imread("CS" + str(m) + ".PNG")

	# Create binary image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

	# Find contours
	(_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 				cv2.CHAIN_APPROX_SIMPLE)

	print("Found %d components." % len(contours))
	# print(contours)
# for i in range(len(contours)):
	# print(len(contours[i]))
	ShapeTypes = SDRFunctions.DetectShapeCS(contours, 2)
	for i in range(ShapeTypes.size):
		print(ShapeTypes[i])

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
	print(np.nonzero(TriangleSDR))
	print(np.nonzero(SquareSDR))
	print(np.nonzero(CircleSDR))
	# Build the SDR for the entire system
	SDR = np.array([])
	if SquareIndex == 0:
		SDR = np.append(SDR, SquareSDR)
	#elif TriangleIndex == 0: 
	#	SDR = np.append(SDR, TriangleSDR)
	elif CircleIndex == 0: 
		SDR = np.append(SDR, CircleSDR)

	if SquareIndex == 1:
		SDR = np.append(SDR, SquareSDR)
	#elif TriangleIndex == 1: 
	#	SDR = np.append(SDR, TriangleSDR)
	elif CircleIndex == 1: 
		SDR = np.append(SDR, CircleSDR)

	#if SquareIndex == 2:
	#	SDR = np.append(SDR, SquareSDR)
	#elif TriangleIndex == 2: 
	#	SDR = np.append(SDR, TriangleSDR)
	#elif CircleIndex == 2: 
	#	SDR = np.append(SDR, CircleSDR)

	NonzeroSDR = np.nonzero(SDR) 
	print(NonzeroSDR)

	# Other Stuff

centroids = []
G = nx.Graph()
tempInt = 0

for c in contours:
	M = cv2.moments(c)
	temp = []
	temp.append(int(M["m10"] / M["m00"]))
	temp.append(int(M["m01"] / M["m00"]))
	centroids.append(temp)
	# Adding nodes to the graph with their attributes
	G.add_node(tempInt, pos = temp)
	tempInt = tempInt + 1

G = Graph_EdgesConstruction(centroids, G, 130.0)

Graph = nx.to_numpy_matrix(G)
print("Graph: ")
print(Graph)

# Number of connected components
number_objects = nx.number_connected_components(G)
print("Number of objects found in the image: ", number_objects)

# Find connected components
objects = sorted(nx.connected_components(G), key = len, reverse=True)

t = 0
centers = []

# New graph for objects
O = nx.Graph()
tmp = 0
for obj in objects:
	tempSum = [0, 0]
	t = 0
	for component in obj:
		t = t + 1
		add(tempSum, G.node[component]['pos'])
	
	centers.append(divide(tempSum, t))
	O.add_node(tmp, pos = centers[tmp])
	tmp = tmp + 1

O = Graph_EdgesConstruction(centers, O, 260.0)

drawGraph(G, 'blue', 'black')
drawGraph(O, 'red', 'white')
plt.imshow(image)
plt.show()
