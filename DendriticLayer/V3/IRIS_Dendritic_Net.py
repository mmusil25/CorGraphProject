"""
This network trains fully connected neural network
on the IRIS dataset using no special ML libraries.

This network is not my original work but has been borrowed as an educational resource.

Code Source:

https://github.com/rianrajagede/iris-python/blob/master/Python/iris_plain_mlp.py
"""
"""
SECTION 1 : Load and setup data for training
"""

import csv
import random
import math
import numpy as np

random.seed(123)

# Load dataset
with open('./iris/iris.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader, None)  # skip header
    dataset = list(csvreader)

# Change string value to numeric
for row in dataset:
    row[4] = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"].index(row[4])
    row[:4] = [float(row[j]) for j in range(len(row))]

# Split x and y (feature and target)
random.shuffle(dataset)
datatrain = dataset[:int(len(dataset) * 0.8)]
datatest = dataset[int(len(dataset) * 0.8):]
train_X = [data[:4] for data in datatrain]
train_y = [data[4] for data in datatrain]
test_X = [data[:4] for data in datatest]
test_y = [data[4] for data in datatest]

"""
SECTION 2 : Build and Train Model
Multilayer perceptron model, with one hidden layer.
input layer : 4 neuron, represents the feature of Iris
hidden layer : 3 neuron, activation using sigmoid
output layer : 3 neuron, represents the class of Iris
optimizer = gradient descent
loss function = Square ROot Error
learning rate = 0.005
epoch = 400
best result = 96.67%
"""


def multi_variate_sigmoid(x):  # Here x is a np array
    return (1 + np.exp(np.sum(x))) ** (-1)


def dendritic_boundary(x):  # Here x is a single valued real number
    alpha_L, alpha_U = 0.5, 0.5
    b_U, b_L = 0, 1
    numerator = (1 + np.exp(alpha_L * (x - b_L))) ** (alpha_L ** (-1))
    denominator = (1 + np.exp(alpha_U * (x - b_U))) ** (alpha_U ** (-1))
    return np.log(numerator / denominator) + b_L


def dendritic_transfer(x):  # Here x is a numpy array
    a_d, c_d, b_d = 1, 0.5, 0.5
    arg1 = c_d * multi_variate_sigmoid(np.multiply(a_d, np.subtract(x, b_d)) + np.sum(x))
    return dendritic_boundary(arg1)


def dendritic_layer(input_set, weights, dendrites, output_features, Den_view, batch_size=None, bias=None):
    """
    batch_size is not used in this implementation.

    """

    # if bias is not None:
    #     bias_np = self.bias.detach().numpy()
    soma_input = np.zeros(dendrites)
    #output = np.zeros((int(batch_size), int(output_features)))
    output = np.zeros(int(output_features))
    #for i in range(batch_size):
    for n in range(output_features):
        for d in range(dendrites):
            # soma_input[d] = np.dot(input_set[d:d + Den_view], weights[n, d].reshape(49, 1))
            soma_input[d] = np.dot(input_set[d:d + Den_view], weights[n, d])
        output[n] = dendritic_transfer(soma_input)
            # if self.bias is not None:
            #     output[i] += bias_np
    return output


def matrix_mul_bias(A, B, bias):  # Matrix multiplication (for Testing)
    C = [[0 for i in range(len(B[0]))] for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C


def vec_mat_bias(A, B, bias):  # Vector (A) x matrix (B) multiplication
    C = [0 for i in range(len(B[0]))]
    for j in range(len(B[0])):
        for k in range(len(B)):
            C[j] += A[k] * B[k][j]
            C[j] += bias[j]
    return C


def mat_vec(A, B):  # Matrix (A) x vector (B) multipilicatoin (for backprop)
    C = [0 for i in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            C[i] += A[i][j] * B[j]
    return C


def sigmoid(A, deriv=False):
    if deriv:  # derivation of sigmoid (for backprop)
        for i in range(len(A)):
            A[i] = A[i] * (1 - A[i])
    else:
        for i in range(len(A)):
            A[i] = 1 / (1 + math.exp(-A[i]))
    return A


# Define parameter
alfa = 0.005
epoch = 400
neuron = [4, 5, 12, 3]  # number of neuron each layer
dendrites = 4
Den_view = 3
# Initiate weight and bias with 0 value
weight = [[0 for j in range(neuron[1])] for i in range(neuron[0])]
weight_2 = [[0 for j in range(neuron[2])] for i in range(neuron[1])]
dendrite_weights = np.subtract(2*np.random.normal(0, 1, [neuron[3], dendrites, int(neuron[2]/dendrites)]),
                               np.full([neuron[3], dendrites, int(neuron[2]/dendrites)], -1))

bias = [0 for i in range(neuron[1])]
bias_2 = [0 for i in range(neuron[2])]

# Initiate weight with random between -1.0 ... 1.0
for i in range(neuron[0]):
    for j in range(neuron[1]):
        weight[i][j] = 2*random.gauss(0, 1) - 1

for i in range(neuron[1]):
    for j in range(neuron[2]):
        weight_2[i][j] = 2*random.gauss(0, 1) - 1

for e in range(epoch):
    cost_total = 0
    for idx, x in enumerate(train_X):  # Update for each data; SGD

        # Forward propagation
        h_1 = vec_mat_bias(x, weight, bias)
        X_1 = sigmoid(h_1)
        h_2 = vec_mat_bias(X_1, weight_2, bias_2)
        X_2 = sigmoid(h_2)
        X_3 = dendritic_layer(X_2, dendrite_weights, dendrites, neuron[3], Den_view)
        #  dendritic_layer(input_set, weights, dendrites, output_features, Den_view, batch_size=None, bias=None)
        # Convert to One-hot target
        target = [0, 0, 0]
        target[int(train_y[idx])] = 1

        # Cost function, Square Root Eror
        eror = 0
        for i in range(3):
            eror += 0.5 * (target[i] - X_3[i]) ** 2
        cost_total += eror

        # Backward propagation
        # Update weight_2 and bias_2 (layer 2)
        delta_3 = []
        for j in range(neuron[3]):
            delta_3.append(-1 * (target[j] - X_3[j]) * X_3[j] * (1 - X_3[j]))

        # for i in range(neuron[1]):
        #     for j in range(neuron[2]):
        #         weight_2[i][j] -= alfa * (delta_3[j] * X_1[i])
        #         bias_2[j] -= alfa * delta_3[j]

        for i in range(neuron[2]):
            for j in range(neuron[3]):
                dendrite_weights[i][j] -= alfa * (delta_3[j] * X_2[i])
                bias_2[j] -= alfa * delta_3[j]

        for i in range(neuron[1]):
            for j in range(neuron[2]):
                weight_2[i][j] -= alfa * (delta_3[j] * X_1[i])
                bias_2[j] -= alfa * delta_3[j]

        # Update weight and bias (layer 1)
        delta_1 = mat_vec(weight_2, delta_3)
        for j in range(neuron[1]):
            delta_1[j] = delta_1[j] * (X_1[j] * (1 - X_1[j]))

        for i in range(neuron[0]):
            for j in range(neuron[1]):
                weight[i][j] -= alfa * (delta_1[j] * x[i])
                bias[j] -= alfa * delta_1[j]

    cost_total /= len(train_X)
    if (e % 100 == 0):
        print("Epoch" , e/100, " out of ", epoch/100)
        print("Epoch cost: ", cost_total)

"""
SECTION 3 : Testing
"""

res = matrix_mul_bias(test_X, weight, bias)
res_2 = matrix_mul_bias(res, weight_2, bias)

# Get prediction
preds = []
for r in res_2:
    preds.append(max(enumerate(r), key=lambda x: x[1])[0])

# Print prediction
print("Predictions: ", preds)

# Calculate accuration
acc = 0.0
for i in range(len(preds)):
    if preds[i] == int(test_y[i]):
        acc += 1
print("Network Accuracy: ", acc / len(preds) * 100, "%")

