# Cortical-Like Self Organizing Map 
### CorGraph Design Project - Portland State University
### August 8th, 2018

The goal of this project is to create a self organizing map that mimics certain functions of a layer in the neocortex. This work
is based on the Spatial Pooler project by [Numenta](numenta.com) (paper included in this directory) which also simulates a cortical 
sheet but uses binary inputs. This is an unsuperized machine learning algorithm that reduces dimensionality in the same way the 
standard self organizing maps do but it also prepares the data for time series prediction.

### Version 0.1 Features

#### 1. Multiple neurons activate for a given input

This allows for the sparse distibuted representation (SDR) used in the Numenta spatial pooler. SDRs are more robust to noise then one hot encoding and can represent
a larger input space.

#### 2. Receptive fields of the neurons are limited to a hypercube that is a subset of the input space. 
The centroid of this hypercube is equal to the weights of the neurons and thus the centroid moves along with the weights to reach more appropriate regions of input.
Overlap may exist between receptive fields but this overlap can be controlled by changing the hypercube side length.

#### 3. Local competition betweens neurons that have the same input vector.  

This will ensure that only a certain percentage of neurons activate for a given input. Neurons will compete within their neighborhood and only those within a certain
percentile will be allowed to activate. The standard SOM weight adjustments that proceed this competition satisfy the Hebbian rule of "Neurons that fire together
wire together". 
