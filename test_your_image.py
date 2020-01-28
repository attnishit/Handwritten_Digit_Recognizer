import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np 
import cv2

#loading the trained model
model = tf.keras.models.load_model('model.h5')

#adding the image
file = cv2.imread('test_images/download2.png')
file = cv2.resize(file, (28, 28))
file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
file=file/255.
plt.imshow(file)
plt.show()
file = file.reshape(( 28, 28,1))

x = image.img_to_array(file)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print(classes)