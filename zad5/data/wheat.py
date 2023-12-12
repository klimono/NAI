import pandas as pd
import tensorflow as tf
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
    3. layer - 4 neurons using "softmax" activation function
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

print()
"""
Show the loss value and metrics for the model in test mode.
"""
model.evaluate(X_test, y_test)