import tensorflow as tf 
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np 
import cv2
#loading the dataset into training and test
#training_data=60000 images
#test_data=10000 images
mnist=tf.keras.datasets.mnist
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()
#reshaping images
test_images=test_images.reshape(10000,28,28,1)
test_images=test_images/255.0

#loading the trained model
model = tf.keras.models.load_model('model.h5')

#describe the information about different layers in neural network
model.summary()

#evaluating the result on test data
test_loss,test_acc=model.evaluate(test_images,test_labels)
#print accuracy
print(test_acc)
