"""
The program uses a neural network to predict the class of one of ten objects:
0: T-shirt/top
1: Spodnie
2: Sweter
3: Sukienka
4: Kurtka
5: Sandałki
6: Koszula
7: Sneakers
8: Torebka
9: Buty

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

Preparing the environment:
    libraries to install: tensorflow, numpy, pandas
    to install  use command: "pip install <name of library >"
"""

import pandas as pd
import numpy as np
import math
import tensorflow as tf

"""
Read data from the file.
"""
read_data = pd.read_csv("data/fashion-mnist.csv").to_numpy()

"""
Randomly shuffle the dataset.
"""
np.random.shuffle(read_data)
X = read_data[:, 1:] / 255.0
y = read_data[:, 0]


def split_data_train_validation_test(size, training_size=0.8, validation_size=0.1):
    """
    Function that splits a set of images into a validation, training, and testing set.

    Parameters:
        size (int): Number of images.
        training_size (float): Size of training set.
        validation_size (float): Size of validation set.
    Returns:
        int: Size of training, validation and test set.
    """
    training = math.floor(training_size * size)
    validation = math.floor(validation_size * size)
    test = size - training - validation

    return training, validation, test


"""
Load final sets sizes before dividing whole dataframe.
"""
training_size, validation_size, test_size = split_data_train_validation_test(read_data.shape[0])

"""
Divide dataframe between training, test and validation sets.
"""
X_training, y_training = X[:training_size], y[:training_size]
X_validation, y_validation = X[training_size:validation_size + training_size], y[
                                                                               training_size:validation_size + training_size]
X_test, y_test = X[training_size + validation_size:], y[training_size + validation_size:]


"""
Declare neural network model.
"""
model = tf.keras.Sequential()

"""
Add neural network layers to model.
"""
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(784, activation='elu'))
model.add(tf.keras.layers.Dense(128, activation='elu'))
model.add(tf.keras.layers.Dense(64, activation='elu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

"""
Define compilation model with specified optimizer, loss function and metrics for created neural network.
"""
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
Train neural network on training set with defined epochs count and batch size.
Validation is performed on prepared part of set.
"""
model.fit(X_training, y_training, epochs=30, batch_size=16,
          validation_data=(X_validation, y_validation))

"""
Show the loss value and metrics for the model in test mode.
"""
model.evaluate(X_test, y_test)