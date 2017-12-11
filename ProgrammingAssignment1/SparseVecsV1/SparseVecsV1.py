#================================================
#Python Code for processing an adjacency matrix, creating sparse representations
#for each vertex, and representations edges by creating vectors for each row
#which are the concatenation of the sparse representations for each vertex.
#Created for the purposes of the CorGraph research group led by Dr. Dan Hammerstrom
#Written by Mark Musil on November 23, 2017
#==============================================
import sparsefuncs as sf
import csv

#Configuration of user choice of matrix input method 

options = {1 : sf.inputFromFile, 2 : sf.manualinput}

#Get sparse representation parameters: Sparsity, Word Width

while True: #Get sparseness density
    sparse = float(input("Enter the Sparsity percentage as a decimal between 0 and 1\n"))
    if(((sparse<=0)|(sparse>1))|(str(sparse)=='')):
        print('The value you entered is not valid (0 < sparseness < 1)\n')
    else: 
        break
print("Sparsity percentage = " + str(sparse*100) + "%\n")

while True:  #Get word width
    width = int(input("Enter the integer width for a vertice's binary representation\n"))
    if ((width<=2)):
        print('The value you entered is not valid (width > 2) ')
    else:
        break
print("Word width = " + str(width) + " bits\n")

while True: #Get number of vertices in the adjacency matrix
    vertices = int(input("Enter the number of vertices in your adjacency Matrix\n"))
    if((vertices<2)|(str(vertices)=='')):
        print('That is not an acceptable number (must have at least 2 vertices)')
    else:
        break
while True:
    inputchoice = int(input('Enter 1 for file entry and 2 for manual adjacency matrix entry\n'))
    if (((inputchoice != 1)&(inputchoice != 2))|(str(inputchoice)=='')):
        print('That is not a valid entry, pick 1 or 2')
    else:
        break
    
ADJ = options[inputchoice](vertices) #Invoke the chosen matrix entry method
Sparsevecs = sf.SparseRepGen(vertices, sparse, width, ADJ) #Generate sparse reps for each vertex
print 'Your final set of vectors will be written out to a csv file.\n'
outputFileName = raw_input('Enter the desired name of your output file with no spaces and without the .csv extension.\n')
outputFileName = outputFileName + '.csv'
ConcatMat = sf.ConcatMatGen(vertices, Sparsevecs, ADJ, outputFileName) #Generate the matrix of concatenated sparse reps
 
#Writes the result to a file.
with open(outputFileName, 'wb') as csvfile:
    outputWriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(vertices): 
        outputWriter.writerow(ConcatMat[i])
    

