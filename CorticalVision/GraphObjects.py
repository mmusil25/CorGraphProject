import numpy as np
import imutils
import cv2
import random
import itertools
import networkx as nx
import keras
import sys
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model

model = load_model('/home/mark/Documents/CorGraphProjectGit/CorticalVision/CnnTrainer/TrainedModel5')
shapeDict = {0:'Triangle',1:'Circle',2:'Square'}

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


# Read image
image = cv2.imread("TCS10.PNG")



# Create binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

# Find contours
(_, contours, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Found %d components." % len(contours))
centroids = []
G = nx.Graph()
tempInt = 0
listofImages = []
bufferRect = 0
bufferCNNInput = 30
for c in contours:
	M = cv2.moments(c)
	temp = []
	temp.append(int(M["m10"] / M["m00"]))
	temp.append(int(M["m01"] / M["m00"]))
	x,y,w,h = cv2.boundingRect(c)

	
#======== Cut out the images and prepare them for the CNN

	cropped = image[y-bufferCNNInput:y+h+bufferCNNInput,x-bufferCNNInput:x+w+bufferCNNInput]
	plt.imshow(cropped)
	plt.show()
	resizedImage = cv2.resize(cropped,(32,32))
	reshapedImage = np.reshape(resizedImage,(32,32,3))
	listofImages.append(reshapedImage)
	cv2.rectangle(image,(x-bufferRect,y-bufferRect),(x+w+bufferRect,y+h+bufferRect),(0,255,0),2)

#======== 

	centroids.append(temp)
	# Adding nodes to the graph with their attributes
	G.add_node(tempInt, pos = temp)
	tempInt = tempInt + 1

#======= Pass the shapes to the CNN to find their shape types 

CNNInput = np.stack(listofImages,axis=0)
shapeTypes = model.predict(CNNInput)
for i in range(3):
	max = np.argmax(shapeTypes[i])
	print[shapeDict[max]]

# Need to pass these finds out
#===========

G = Graph_EdgesConstruction(centroids, G, 130.0)
Graph = nx.to_numpy_matrix(G)
print("Graph: ")
print(Graph)

#==========
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
plt.savefig('TCS10Augmented',dpi=1000,bbox_inches='tight',pad_inches=0.1)
plt.show()
