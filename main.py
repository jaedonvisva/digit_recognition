import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

model = tf.keras.models.load_model('digits.model')



for x in range(1, 8):
  img = cv2.imread(f'{x}.png')[:, :, 0]
  img = np.invert(np.array([img]))
  prediction = model.predict(img)
  print(np.argmax(prediction))
  plt.imshow(img[0], cmap=plt.cm.binary)
  plt.show()