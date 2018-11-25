
# HTM Spatial Pooler. This code implements the HTM spatial
# pool described by Numenta in order to extract order in an
# unsupervised fashion. It is an unsupervised learning system
# and is described in greater detail at numenta.org
#
# The input is a sparse distributed representations of a graph which
# was generating by Aakanksha Mathuria <aakanksha.mathuria@gmail.com>

import numpy
import nupic
import os

from nupic.algorithms.spatial_pooler import SpatialPooler

# Section 1: Converting indice based SDRs into binary vectors.

EncodingWidth, ActiveBits, SpatialPoolerWidth = 1336-1, 122, 2048

Graph1EncodedIndices = numpy.array([3, 4, 6, 13, 32, 33, 37, 38, 42, 69, 83, 93,
                        104, 108, 110, 124, 127, 132, 143, 146, 147,
                        153, 154, 163, 187, 188, 192, 233, 258, 260,
                        277, 296, 298, 305, 361, 375, 424, 437, 440,
                        466, 503, 527, 566, 579, 580, 608, 609, 669,
                        700, 722, 751, 822])
Graph2EncodedIndices = numpy.array([3, 4, 6, 13, 32, 33, 38, 69, 83, 94, 107,
                        128, 131, 136, 146, 147, 153, 163, 188, 233,
                        281, 295, 298, 305, 361, 375, 428, 437, 440,
                        466, 503, 528, 570, 579, 580, 608, 609, 670,
                        704, 722, 751, 825])
Graph1Binary = numpy.zeros(EncodingWidth)
Graph2Binary = numpy.zeros(EncodingWidth)

for i in range(Graph1EncodedIndices.size):
    Graph1Binary[Graph1EncodedIndices[i]] = 1

for i in range(Graph2EncodedIndices.size):
    Graph2Binary[Graph2EncodedIndices[i]] = 1



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
  numActiveColumnsPerInhArea=40.0,
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

activeColumns1 = numpy.zeros(SpatialPoolerWidth)
activeColumns2 = numpy.zeros(SpatialPoolerWidth)


dotproduct = numpy.dot(Graph1Binary, Graph2Binary)/(numpy.linalg.norm(Graph1Binary)*numpy.linalg.norm(Graph2Binary))


# Find percent overlap of old vectors.

print('Percent Overlap of the Input Vectors: ' + str(round(dotproduct,5)*100) +'%' )

# Run the spatial pooling function

sp.compute(Graph1Binary, True, activeColumns1)
activeColumnIndices1 = numpy.nonzero(activeColumns1)[0]
print('Spatial Pooler Output for Graph 1: ' + str(activeColumnIndices1))

sp.compute(Graph2Binary, True, activeColumns2)
activeColumnIndices2 = numpy.nonzero(activeColumns2)[0]
print('Spatial Pooler Output for Graph 2:' + str(activeColumnIndices2))

# find percent overlap of spatially pooled vectors.


dotproduct = numpy.dot(activeColumns1, activeColumns2)/(numpy.linalg.norm(activeColumns1)*numpy.linalg.norm(activeColumns2))

print('Percent Overlap of the SP Output Vectors: ' + str(round(dotproduct,5)*100) +'%' )
