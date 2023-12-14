"""
The program uses a neural network to predict the class of one of ten objects:
1. Airplane
2. Car
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Authors:
    Oliwier Kossak s22018
    Daniel Klimowski S18504

Preparing the environment:
    libraries to install: tensorflow
    to install  use command: "pip install <name of library >"
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10

"""
Load cifar10 as train and test datasets.
"""
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

"""
Set validation set size
"""
validation_size = 0.2
validation_images_size = int(train_images.shape[0] * 0.2)


"""
Divide dataframe between training, test and validation sets.
"""
X_training = train_images[validation_images_size:]
y_training = train_labels[validation_images_size:]

X_validation = train_images[:validation_images_size]
y_validation = train_labels[:validation_images_size]

X_test = test_images
y_test = test_labels

"""
Declare neural network model.
"""
model = tf.keras.Sequential()

"""
Add neural network layers to model.
"""
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
model.summary()

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