import tensorflow as tf 
import matplotlib.pyplot as plt


#stopping the training when reaching 99% accuracy
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training=True
mnist=tf.keras.datasets.mnist

#creating instance of class
callbacks=mycallback()

#loading the dataset into training and test
#training_data=60000 images
#test_data=10000 images
(training_images,training_labels),(test_images,test_labels)=mnist.load_data()

#display some images
plt.imshow(training_images[0])

plt.show()

#reshaping images to be fitted in neural networks
training_images=training_images.reshape(60000,28,28,1)
test_images=test_images.reshape(10000,28,28,1)
training_images=training_images/255.0
test_images=test_images/255.0

#using convolutional neural network and then applying softmax classification to classify the type of image 
model=tf.keras.Sequential([tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128,activation='relu'),
	tf.keras.layers.Dense(10,activation='softmax')
    ])

#applying adam optimizer and minimizing the cost
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#describe the information about different layers in neural network
model.summary()

#fitting the model on training data 
model.fit(training_images,training_labels,epochs=15,callbacks=[callbacks])
model.save("model.h5")
print("Saved model to disk")