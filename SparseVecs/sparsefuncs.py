# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 01:14:03 2017
This module defines several integral functions for the SparseVec program.
@author: Mark Musil
"""
import numpy as np
import csv

#------Function supporting manual entry of an adjacency matrix------
def manualinput(vertices):  
    print("Your adjacency matrix must have " + str(vertices) + " rows and columns when entered\n")
    ADJ = []
    row = []
    for i in range(vertices):
        row.append(0)
    for k in range(vertices):
        for i in range(vertices):
            while True:
                row[i]=((int(input("Input entry "+str(i)+" of row "+str(k)+"\n"))))
                if(((row[i]!=1)&(row[i]!=0))|(str(row[i])=='')):
                    print('That entry is invalid please re-enter\n')
                else:
                    break
        ADJ.append(row[:]) 

    return ADJ
#----------------Function for randomly generating sparse representations--------
def genList(sparse, width):
    count = round(sparse*width) # Calculates # of 1s
    idxs = np.arange(width)     # generates linear array
    np.random.shuffle(idxs)     # shuffle
    idxs = idxs[:count]         # slice first count elements
    ans = np.zeros((width,), dtype=np.int) # fill with 0's
    ans[idxs] = 1                        # Turn on count 1's
    return ans
    
#-------------Function for getting input from a file--------------

def inputFromFile(vertices, delimit=' '):
    inputFileName = input('Enter the name of your input file with no spaces and without the .csv extension.\n')
    inputFileName = inputFileName + '.csv'		
    with open(inputFileName, 'rb') as csvfile:
        matreader = csv.reader(csvfile, delimiter=',',quotechar='|')
        ans = []
        try:
            for row in matreader:
                elements = map(int, row)
                ans = ans + [elements,]
        except ValueError:
            None ##Code goes here
    return np.array(ans)
#    ADJ=[]
#     row=[]
#     for i in range(vertices):
#         row.append(0)
#     name = input("input the name of your csv file\n")
#     with open(name) as f:
#         for line in f:
#             for i in range(len(line)):            
#                 if line[i].isdecimal():
#                     if((int(line[i])!=0)&(int(line[i])!=1)):
#                         print("Check entry "+str(i)+ " of line "+str(i)+" it is not 0 or 1")
#                         return 1 #Failure of value to be 0 or 1
#                     row[i]=int(line[i])
#             ADJ.append(row[:])  #Add this new row to ADJ                    
#     return ADJ;
#==============================================================================
#------------Function which utilizes genList() to build a dynamic number of
    #sparse representations for the given parameters.
def SparseRepGen(vertices, sparse, width, ADJ):
    nodeRep=[]
    Sparsevecs=[]
    for i in range(vertices):
        nodeRep.append(0)
    for k in range(vertices):
        nodeRep=genList(sparse,width).tolist()
        Sparsevecs.append(nodeRep[:])      
    print('Your adjacency matrix is:\n')
    for i in range(vertices):
        print(ADJ[i])
    print('\nYour sparse representation for each vertex is:')
    for i in range(vertices):
        print('Vertex '+str(i)+':')
        print(Sparsevecs[i])
    return Sparsevecs

#-------------Function for generating vector of each row--------------
#This function navigates each row and generates a vector which is 
#the concatenation of the sparse representations for the vertices
#connected to a given vertex.

def ConcatMatGen(vertices, Sparsevecs, ADJ,outputFileName):
    
    concatVec=[]
    outputToCNN=[]


    for i in range(vertices): #Loop for each row
        concatVec.append(Sparsevecs[i]) #Always put the ith sparse rep. as the base entry to the vector
        SlicedADJ = ADJ[i]    #Grab the ith row from ADJ for processing
        for j in range(vertices):
            if SlicedADJ[j]==1:  #If the jth entry is 1, add to concatenated vector
                concatVec.append(Sparsevecs[j])   
        outputToCNN.append(concatVec)  #Once all columns are processed, add the concatenated vector
            #to the master list of output vectors for the next stage.
        concatVec=[]
                
    for i in range(vertices):
        print('The concatenated vector of sparse representations for vertex '+str(i)+':')
        print(outputToCNN[i])
    return outputToCNN
