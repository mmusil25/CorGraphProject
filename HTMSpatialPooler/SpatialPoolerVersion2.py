
# HTM Spatial Pooler. This code implements the HTM spatial
# pool described by Numenta in order to extract order in an
# unsupervised fashion. It is an unsupervised learning system
# and is described in greater detail at numenta.org
#
# The input is a sparse distributed representations of a graph which
# was generating by Aakanksha Mathuria <aakanksha.mathuria@gmail.com>

import numpy as np
import nupic
import os

from nupic.algorithms.spatial_pooler import SpatialPooler

# Section 1: Converting indice based SDRs into binary vectors.

EncodingWidth, ActiveBits, SpatialPoolerWidth = 590, 122, 180

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
CGraphEncodedIndices = np.array([2, 42, 83, 104])


TCSGraph0EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph1EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph2EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph3EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 201, 224, 242, 282, 323, 344])
TCSGraph4EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph5EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph6EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph7EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph8EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph9EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph10EncodedIndices = np.array([  2,  42,  81, 104, 122, 162, 203, 224, 244, 282, 323, 341])
TCSGraph11EncodedIndices = np.array([  2,  42,  81, 104, 122, 162, 203, 224, 244, 282, 323, 341])
TCSGraph12EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph13EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph14EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph15EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])
TCSGraph16EncodedIndices = np.array([  4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344]) 
TCSGraph17EncodedIndices = np.array([  2,  42,  83, 104, 122, 162, 201, 224, 244, 282, 323, 341]) 
TCSGraph18EncodedIndices = np.array([4,  42,  83, 101, 122, 162, 201, 224, 242, 282, 323, 344])
TCSGraph19EncodedIndices = np.array([4,  42,  83, 101, 122, 162, 203, 224, 242, 282, 321, 344])


TCSGraph0Binary = np.zeros(EncodingWidth)
TCSGraph1Binary = np.zeros(EncodingWidth)
TCSGraph2Binary = np.zeros(EncodingWidth)
TCSGraph3Binary = np.zeros(EncodingWidth)
TCSGraph4Binary = np.zeros(EncodingWidth)
TCSGraph5Binary = np.zeros(EncodingWidth)
TCSGraph6Binary = np.zeros(EncodingWidth)
TCSGraph7Binary = np.zeros(EncodingWidth)
TCSGraph8Binary = np.zeros(EncodingWidth)
TCSGraph9Binary = np.zeros(EncodingWidth)
TCSGraph10Binary = np.zeros(EncodingWidth)
TCSGraph11Binary = np.zeros(EncodingWidth)
TCSGraph12Binary = np.zeros(EncodingWidth)
TCSGraph13Binary = np.zeros(EncodingWidth)
TCSGraph14Binary = np.zeros(EncodingWidth)
TCSGraph15Binary = np.zeros(EncodingWidth)
TCSGraph16Binary = np.zeros(EncodingWidth)
TCSGraph17Binary = np.zeros(EncodingWidth)
TCSGraph18Binary = np.zeros(EncodingWidth)
TCSGraph19Binary = np.zeros(EncodingWidth)


for i in range(TCSGraph0EncodedIndices.size):
    TCSGraph0Binary[TCSGraph0EncodedIndices[i]] = 1

for i in range(TCSGraph1EncodedIndices.size):
    TCSGraph1Binary[TCSGraph1EncodedIndices[i]] = 1

for i in range(TCSGraph2EncodedIndices.size):
    TCSGraph2Binary[TCSGraph2EncodedIndices[i]] = 1

for i in range(TCSGraph3EncodedIndices.size):
    TCSGraph3Binary[TCSGraph3EncodedIndices[i]] = 1

for i in range(TCSGraph4EncodedIndices.size):
    TCSGraph4Binary[TCSGraph4EncodedIndices[i]] = 1

for i in range(TCSGraph5EncodedIndices.size):
    TCSGraph5Binary[TCSGraph5EncodedIndices[i]] = 1

for i in range(TCSGraph6EncodedIndices.size):
    TCSGraph6Binary[TCSGraph6EncodedIndices[i]] = 1

for i in range(TCSGraph7EncodedIndices.size):
    TCSGraph7Binary[TCSGraph7EncodedIndices[i]] = 1

for i in range(TCSGraph8EncodedIndices.size):
    TCSGraph8Binary[TCSGraph8EncodedIndices[i]] = 1

for i in range(TCSGraph9EncodedIndices.size):
    TCSGraph9Binary[TCSGraph9EncodedIndices[i]] = 1

for i in range(TCSGraph10EncodedIndices.size):
    TCSGraph10Binary[TCSGraph10EncodedIndices[i]] = 1

for i in range(TCSGraph11EncodedIndices.size):
    TCSGraph11Binary[TCSGraph11EncodedIndices[i]] = 1

for i in range(TCSGraph12EncodedIndices.size):
    TCSGraph12Binary[TCSGraph12EncodedIndices[i]] = 1

for i in range(TCSGraph13EncodedIndices.size):
    TCSGraph13Binary[TCSGraph13EncodedIndices[i]] = 1

for i in range(TCSGraph14EncodedIndices.size):
    TCSGraph14Binary[TCSGraph14EncodedIndices[i]] = 1

for i in range(TCSGraph15EncodedIndices.size):
    TCSGraph15Binary[TCSGraph15EncodedIndices[i]] = 1

for i in range(TCSGraph16EncodedIndices.size):
    TCSGraph16Binary[TCSGraph16EncodedIndices[i]] = 1

for i in range(TCSGraph17EncodedIndices.size):
    TCSGraph17Binary[TCSGraph17EncodedIndices[i]] = 1

for i in range(TCSGraph18EncodedIndices.size):
    TCSGraph18Binary[TCSGraph18EncodedIndices[i]] = 1

for i in range(TCSGraph19EncodedIndices.size):
    TCSGraph19Binary[TCSGraph19EncodedIndices[i]] = 1


#=====================Circle and Square ===============================

CSGraphEncodedIndices = np.array([4,  42,  83, 101, 122, 162, 203, 224])

CSGraphBinary = np.zeros(EncodingWidth)

for i in range(CSGraphEncodedIndices.size):
    CSGraphBinary[CSGraphEncodedIndices[i]] = 1

sp = SpatialPooler(
  # How large the input encoding will be.
  inputDimensions=(EncodingWidth),
  # How many mini-columns will be in the Spatial Pooler.
  columnDimensions=(SpatialPoolerWidth),
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

# Array which contains the output of the spatial pooler

activeColumns1 = np.zeros(SpatialPoolerWidth)
activeColumns2 = np.zeros(SpatialPoolerWidth)
activeColumns3 = np.zeros(SpatialPoolerWidth)
activeColumns4 = np.zeros(SpatialPoolerWidth)


# Compare overlap of the input SDRs
#dotproduct = numpy.dot(Graph1Binary, Graph2Binary)/(numpy.linalg.norm#(Graph1Binary)*numpy.linalg.norm(Graph2Binary))

# Find percent overlap of old vectors.

#print('Percent Overlap of the Input Vectors: ' + str(round(dotproduct,5)*100) +'%' )

# Run the spatial pooling function for Triangle Square Circle

for i in range(10):
	sp.compute(TCSGraph0Binary, True, activeColumns1)
	activeColumnIndices1 = np.nonzero(activeColumns1)[0]

	sp.compute(TCSGraph10Binary, True, activeColumns2)	
	activeColumnIndices2 = np.nonzero(activeColumns2)[0]

print('Spatial Pooler Output for Triangle Square Circle: ' + str(activeColumnIndices2))


# Run the spatial pooling function for the Circle Square

for m in range(20):
	sp.compute(CSGraphBinary, True, activeColumns3)
	activeColumnIndices3 = np.nonzero(activeColumns3)[0]

print('Spatial Pooler Output for Circle Square:' + str(activeColumnIndices3))

# Run the spatial pooling function for Triangle Square


# Run the spatial pooling function for Triangle Circle


# find percent overlap of spatially pooled vectors vs input vectors.

dotproduct = np.dot(TCSGraph0Binary, CSGraphBinary)/(np.linalg.norm(TCSGraph0Binary)*np.linalg.norm(CSGraphBinary))

print('Percent Overlap of the SP Input Vectors (Triangle Square Circle and Circle Square): ' + str(round(dotproduct,5)*100) +'%' )

dotproduct = np.dot(activeColumns1, activeColumns3)/(np.linalg.norm(activeColumns1)*np.linalg.norm(activeColumns3))

print('Percent Overlap of the SP Output Vectors(Triangle Square Circle and Circle Square): ' + str(round(dotproduct,5)*100) +'%' )

#Run just the circle through the spatial pooler

CGraphEncodedIndices = np.array([2, 42, 83, 104])

CGraphBinary = np.zeros(EncodingWidth)

for i in range(CGraphEncodedIndices.size):
    CGraphBinary[CGraphEncodedIndices[i]] = 1

sp.compute(CGraphBinary, True, activeColumns4)
activeColumnIndices4 = np.nonzero(activeColumns4)[0]

dotproduct = np.dot(activeColumns1, activeColumns4)/(np.linalg.norm(activeColumns1)*np.linalg.norm(activeColumns4))

print('Percent Overlap of the SP Output Vectors(Circle alone and Triangle Square Circle): ' + str(round(dotproduct,5)*100) +'%' )

dotproduct = np.dot(activeColumns3, activeColumns4)/(np.linalg.norm(activeColumns3)*np.linalg.norm(activeColumns4))

print('Percent Overlap of the SP Output Vectors(Circle alone and Circle Square): ' + str(round(dotproduct,5)*100) +'%')


