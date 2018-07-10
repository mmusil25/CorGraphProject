# CNN Training script

# Trying out a CNN on the CIFAR10 set.
# Mark Musil B.S.E.E. 10-May-2018
import keras
import numpy
import matplotlib.pyplot as plt
import TrainingTestValBuilder as sets
import datetime

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,PReLU, BatchNormalization

keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

#Global Variables

inputSideLength = 32
t = datetime.time(1, 2, 3,4)
d = datetime.date.today()
#dt = datetime.datetime.combine(d, t)
dt = '6'

(train_data,train_labels_one_hot),(test_data,test_labels_one_hot),(valid_data,valid_labels_one_hot)= sets.buildSets(500,inputSideLength)


keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)


#print(train_data.shape,train_labels_one_hot.shape)
#print(train_labels_one_hot)

def createModel(inputSideLength):
    input_shape = (inputSideLength,inputSideLength,3)
    nClasses = 3
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nClasses, activation='softmax'))

    return model

model1 = createModel(inputSideLength)
batch_size = 70
epochs = 20
model1.summary()
model1.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, 
                   validation_data=(test_data, test_labels_one_hot))

model1.evaluate(valid_data, valid_labels_one_hot)
model1.save('TrainedModel'+str(dt))

# Loss Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['loss'], 'r', linewidth=3.0)
plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.title('Loss Curves' + str(dt), fontsize=16)
plt.savefig('Loss Curves' + str(dt),dpi=1000,bbox_inches='tight',pad_inches=0.1)


# Accuracy Curves
plt.figure(figsize=[8, 6])
plt.plot(history.history['acc'], 'r', linewidth=3.0)
plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs ', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves' + str(dt), fontsize=16)
plt.savefig('Accuracy Curves' + str(dt),dpi=1000,bbox_inches='tight',pad_inches=0.1)
plt.show()
