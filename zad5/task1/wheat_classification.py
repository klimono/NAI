"""
Program uses neural network to predict wheat species based on seed measurements of different wheat varieties.

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

Preparing the environment:
    libraries to install: tensorflow, numpy, pandas
    to install  use command: "pip install <name of library >"
"""

import pandas as pd
import tensorflow as tf
import numpy as np
import math

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
Read data from the file and random sample it.
"""
df = pd.read_csv('data/wheat.csv')
df = df.sample(frac=1)

"""
Load final sets sizes before dividing whole dataframe.
"""
training_size, validation_size, test_size = split_data_train_validation_test(df.shape[0])

"""
Divide data and labels.
"""
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

"""
Divide dataframe between training, validation and test datasets.
"""
X_training, y_training = X[:training_size], y[:training_size]
X_validation, y_validation = X[training_size:validation_size + training_size], y[training_size:validation_size + training_size]
X_test, y_test = X[training_size + validation_size:], y[training_size + validation_size:]


"""
Define neural network model compared from three layers:
    1. layer - 256 neurons using "relu" activation function     
    2. layer - 128 neurons using "relu" activation function     
    3. layer - 3 neurons using "softmax" activation function
"""
model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation = 'relu'),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(3, activation='softmax')
])

"""
Define compilation model with specified optimizer, loss function and metrics for created neural network.
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
Train neural network on training set with defined epochs count and batch size.
Validation is performed on prepared part of set.
"""
model.fit(X_training, y_training, epochs=30, batch_size=16,
                        validation_data=(X_validation, y_validation))

"""
Define second neural network model compared from three layers:
    1. layer - 525 neurons using "relu" activation function     
    2. layer - 255 neurons using "relu" activation function    
    3. layer - 125 neurons using "relu" activation function     
    4. layer - 3 neurons using "softmax" activation function
"""
model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(525, activation = 'relu'),
        tf.keras.layers.Dense(255, activation = 'relu'),
        tf.keras.layers.Dense(125, activation = 'relu'),
        tf.keras.layers.Dense(3, activation='softmax')
])

"""
Define compilation model with specified optimizer, loss function and metrics for created neural network.
"""
model2.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print()

"""
Train neural network on training set with defined epochs count and batch size.
Validation is performed on prepared part of set.
"""
model2.fit(X_training, y_training, epochs=30, batch_size=16,
                        validation_data=(X_validation, y_validation))

"""First model prediction"""
model_pred = model.predict(X_test)
predicted_classes = np.argmax(model_pred, axis=-1)

"""Confusion matrix first model"""
confusion_matrix_model_1 = tf.math.confusion_matrix(y_test, predicted_classes)
print("Confusion matrix first model")
print(confusion_matrix_model_1)

"""Second model prediction"""
model_pred_2 = model.predict(X_test)
predicted_classes_2 = np.argmax(model_pred_2, axis=-1)
print()

"""Confusion matrix second model"""
confusion_matrix_model_2 = tf.math.confusion_matrix(y_test, predicted_classes_2)
print("Confusion matrix second model")
print(confusion_matrix_model_2)

"""
Model evaluation.
"""
print()
print(f"First model evaluate:")
model.evaluate(X_test, y_test)
print()
print(f"Second model evaluate:")
model2.evaluate(X_test, y_test)
