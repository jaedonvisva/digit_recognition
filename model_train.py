import numpy as np
import tensorflow as tf
#get the dataset
mnist = tf.keras.datasets.mnist

#split the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the data (range it from 0 to 1) to make it easier to compute
#only normalize x because y are the labels (numbers 0-9)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#create a feed-forward neural network
model = tf.keras.models.Sequential()
#add a layer: Flatten layer makes data one dimentional (ex. 28 by 28 turns into 784)
#input shape is what goes into the layer; our data is sized 28 x 28
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#dense layer makes all neurons connected to the previous and next layer
#units is the number of neurons in the layer
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=107)

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')