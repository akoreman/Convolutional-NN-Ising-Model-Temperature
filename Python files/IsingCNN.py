# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import InputOutput as IO

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D

from scipy.optimize import curve_fit

"""
Callback for keras to save the weight matrix for every epoch.
"""
class WeightweightHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        self.weights.append(self.model.layers[-1].get_weights()[0])
        
"""
The Tanh function to weightFit to the data.
"""
def TanhFunction(x, a, b, c, beta):
    return ( a*np.tanh(c*(x - beta)) - b)

"""
Load the data set and label list.
"""
lattices = IO.ReadCSV('LatticesCNN.csv')
temperatureLabels = IO.ReadCSV('TemperatureLabelsCNN.csv')

"""
Convert the labels from temperatures to inverse temperatures.
"""
temperatureLabels = 1/temperatureLabels

trainLattices, testLattices, trainLabels, testLabels = train_test_split(lattices,temperatureLabels,train_size = 0.9, test_size = 0.1)

numberClasses = 100
batchSize = len(trainLattices)
numberEpochs = 500

latticeSize = int(len(lattices[0])**(1/2))

trainLattices = trainLattices.reshape(trainLattices.shape[0], latticeSize, latticeSize, 1)
testLattices = testLattices.reshape(testLattices.shape[0], latticeSize, latticeSize, 1)

"""
Construct the discretized inverse temperatures to use as classes in the classifying.
"""
minTemperature = np.min(temperatureLabels)
maxTemperature = np.max(temperatureLabels)

temperatureBins = np.linspace(minTemperature, 1, numberClasses)

"""
Convert the labels to a one-hot representation corresponding with the discretized temperatures defined aboves.
"""
traindig = np.digitize(trainLabels,temperatureBins)
testdig = np.digitize(testLabels,temperatureBins)

trainLabels = keras.utils.to_categorical(traindig - 1, numberClasses)
testLabels = keras.utils.to_categorical(testdig - 1, numberClasses)

"""
Define the network using Keras sequential layers.
"""
model = Sequential()
model.add(Conv2D(3, kernel_size=(25, 25),
                 strides = (13,13),
                 activation='relu',
                 input_shape=(latticeSize,latticeSize,1)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(numberClasses, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

"""
Use the callback defined above to store the weight matrix of each epoch.
"""
weightHistory = WeightweightHistory()

"""
Train and evaluate the network.
"""
model.fit(trainLattices, trainLabels,
          batch_size=batchSize,
          epochs=numberEpochs,
          verbose=2 ,
          validation_data=(testLattices, testLabels),
          callbacks = [weightHistory])

modelScore = model.evaluate(testLattices, testLabels, verbose=2)

print('Test loss:', modelScore[0])
print('Test accuracy:', modelScore[1])

"""
Plot the matrix of the fully connected layer for the last epoch.
"""
weightMatrix = model.layers[-1].get_weights()[0]

plt.imshow(weightMatrix, cmap = 'Greys')
plt.savefig("Weights.pdf")
plt.show()

"""
Plot the sums of the weights for the fully connected layer.
"""
sumWeights = np.sum(weightMatrix, axis = 0)

"""
Fit the tanh function to the summed weights and give the estimate for the critical temperature with the error given by the covariance of the fit.
"""
weightFit = curve_fit(TanhFunction, temperatureBins, sumWeights, p0 = [0.5,-6,1,0.5], bounds = [[-1*np.inf,-1*np.inf,-1*np.inf,0],[np.inf,np.inf,np.inf,1]] )

weightFitParameters = weightFit[0]
weightFitParameterErrors = np.sqrt(np.diag(weightFit[1]))

print(str(weightFitParameters[3]) + " +/- " + str(weightFitParameterErrors[3]))

"""
Plot the sums of the weights for the fully connected layer with the weightFit.
"""
plt.plot(temperatureBins, sumWeights, 'bo', markersize = 3)
plt.xlabel("1/T [dimensionless units]")
plt.ylabel("Summed weights [arbitrary units]")
plt.savefig("SummedWeights.pdf")

plt.plot(temperatureBins, TanhFunction(temperatureBins, *weightFitParameters), markersize = 3)
plt.legend(['Summed weights', r'weightFit: $a \times \tanh(c(x - \beta_C)) - b$'], loc='upper right')
plt.savefig("WeightFit.pdf")
plt.show()

"""
Calculate the estimate for the critical temperature for each 10th epoch to see the development of the estimate vs the number of numberEpochs.
"""
weightFitList = []

for lattices in weightHistory.weights[::10]:   
    sumWeights = np.sum(lattices, axis = 0)
    weightFit = curve_fit(TanhFunction, temperatureBins, sumWeights, p0 = [0.5,-6,1,0.5], bounds = [[-1*np.inf,-1*np.inf,-1*np.inf,0],[np.inf,np.inf,np.inf,1]] )
    weightFitParameters = weightFit[0]
    
    weightFitList.append(weightFitParameters[3])
    
"""
Plot the weights for the fully connected layer for each 50th epoch.
"""
for weightMatrix in weightHistory.weights[::50]:
    plt.imshow(weightMatrix, cmap = 'Greys')
    plt.show()
 
"""
Plot those estimates with the analytical value.
"""
plt.plot(range(20,numberEpochs,10),weightFitList[2:],'bo')
plt.xlabel("numberEpochs")
plt.ylabel(r'$1/T_C$ [dimensionless units]')
plt.hlines(0.44,20,numberEpochs)
plt.savefig("CriticalTemperatureEpochs.pdf")
plt.show