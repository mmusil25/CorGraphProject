#================================================
#
#SparseVecs README
#
#
#Python application for processing an adjacency matrix, creating sparse representations
#for each vertex, and representations edges by creating vectors for each row
#which are the concatenation of the sparse representations for each vertex.
#Created for the purposes of the BICL research group led by Dr. Dan Hammerstrom.
#Written by Mark Musil December 2017
#
#Questions? Email Mark at mmusil@pdx.edu
#
#================================================

The main program is the file SparseVecsV1.py and can be run in any Python environment which supports a command line. 
It has sparsefuncs.py as a dependency and sparsefuncs.py must be contained in the same foldar as SparseVecsV1.py. 

Usage: 

Input can either be manual or via a .csv file. In either case the program will ask for a sparseness, word width, 
and number of vertices. Sparseness (0<sparseness<1) is the percentage of bits set to 1 in your sparse node representations. 
Word width is the size in number of bits of the sparse representations. The number of vertices is the M dimension in your 
MxM adjacency matrix.

Manual Input: You will be allowed to enter each entry manually for processing.
File based input: Add a .csv file to the same directory as SparseVecsV1.py with 1 comma seperating each entry and a
newline seperating rows.

TestFile.csv holds the following example:

0,1,1,0,0
1,0,0,1,1
1,0,0,1,0
0,1,1,0,1
0,1,0,1,0  

The above is a correctly formatted adjacency matrix. 

The program will output a set of sparse representations for each vertex onto the Python Terminal but will also output 
to a file of your naming a set of vectors that are a concatenation of the sparse representations for the ON bits in a
given row. For example the adhacency matrix in TestFile.csv outputs the following set of representations when given a
wordwidth of 10 and sparseness of 0.4.

Your sparse representation for each vertex is:
Vertex 0:
[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
Vertex 1:
[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]
Vertex 2:
[0, 1, 0, 1, 1, 1, 0, 0, 0, 0]
Vertex 3:
[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]
Vertex 4:
[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]


Your final set of concatenated vectors representing the graph will be written out to a csv file. 
For the above example of TestFile.csv the following is produced and the file named 'TestOutput.csv' 
is created in the same directory as SparseVecs1.py.

|[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]| |[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]| |[0, 1, 0, 1, 1, 1, 0, 0, 0, 0]|
|[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]| |[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]| |[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]| |[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]|
|[0, 1, 0, 1, 1, 1, 0, 0, 0, 0]| |[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]| |[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]|
|[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]| |[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]| |[0, 1, 0, 1, 1, 1, 0, 0, 0, 0]| |[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]|
|[1, 1, 0, 0, 1, 0, 0, 1, 0, 0]| |[1, 0, 1, 0, 1, 0, 0, 0, 1, 0]| |[0, 1, 1, 0, 0, 0, 1, 0, 1, 0]|

In the above set of vectors, any entry in the adjacency matrix that was a 0 is not represented. 





