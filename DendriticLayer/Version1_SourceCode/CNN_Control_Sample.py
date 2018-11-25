"""
Name: Mark Musil
Date: November 21, 2018

Project: CorGraph dendritic layer project.

Module: Control CNN without dendritic layer

Description:

This is a simple CNN for MNIST digit recognition based of the design
at https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
This network serves as a control network for the dendritic layer experiments.
In the file " CNN_Dendritic_Layer " The final dense layer is replaced with the
novel dendritically inspired layer. The final dense layer in this network
remains unchanged.

"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
import os
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# # --------------------------------------------
# # Quick figure displaying a sample of the MNIST dataset
# fig = plt.figure()
# for i in range(9):
#         plt.subplot(3, 3, i+1)
#         plt.tight_layout()
#         plt.imshow(X_train[i], cmap='gray', interpolation='none')
#         plt.title("Digit: {}".format(y_train[i]))
#         plt.xticks([])
#         plt.yticks([])
# plt.show()
# # --------------------------------------------

# Reshaping
img_rows, img_cols = 28, 28

if k.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(np.unique(y_train, return_counts=True))

# Set number of categories
num_category = 10

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test  = keras.utils.to_categorical(y_test, num_category)
y_train[0]

# Model Building
model = Sequential()
# Convolutional layer with rectified linear unit activation
# 32 Convolution filters used each of size 3x3
model.add(Conv2D (32, kernel_size=(3, 3),
                  activation='relu',
                  input_shape= input_shape))
# 64 Convolution filters used each of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))
# Choose the best features via pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout improves convergence
model.add(Dropout(0.25))
# Flatten because we only care about classification
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_category, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
batch_size = 128
num_epoch = 10

# Train the model
model_log = model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=num_epoch,
            verbose=1,
            validation_data=(X_test,y_test))

# Basic Metrics
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test Accuracy:', score[1])

# Plot the metrics
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(model_log.history['acc'])
plt.plot(model_log.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2, 1, 2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')

plt.tight_layout()
plt.show()

# Save the model
# Serialize to JSON
model_digit_json = model.to_json()
with open("model_digit.json", "w") as json_file:
    json_file.write(model_digit_json)
# Serialize the weights to HDF5
model.sample_weights("model_digit.h5")
print("Saved model to disk")