# Importing necessary Libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing necessary modules from TensorFlow.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Rescaling


# Reading in the data.
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
	"C:\Users\vsanj\Downloads\archive\chest_xray\train",
	labels = "inferred",
	label_mode = "int",
	class_names = ["NORMAL", "PNEUMONIA"],
	color_mode = "grayscale",
	image_size = (180,180),
	shuffle = True,
	batch_size = 32,
)
ds_test = tf.keras.preprocessing.image_dataset_from_directory(
	"C:\Users\vsanj\Downloads\archive\chest_xray\test",
	labels = "inferred",
	label_mode = "int",
	class_names = ["NORMAL", "PNEUMONIA"],
	#color_mode = "grayscale",
	image_size = (180,180),
	shuffle = True,
	batch_size = 32,
)

# Performing EDA.
# Displaying Random Images from the Dataset:
plt.figure(figsize=(10,10))
for imgs, labels in ds_train.take(1):
	for index in range(9):
		ax = plt.subplot(3,3,index+1)
		plt.imshow(imgs[index].numpy().astype("uint8"))
		plt.title(ds_train.class_names[labels[index]])
		plt.axis("off")
plt.show()

# Constructing the Model.
model = Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "same", input_shape = (180,180,3)))
model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2), padding = "same"))
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "same"))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(2))

# Compiling the Model.
model.compile(optimizer = "adam", 
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics = ["accuracy"]
              )

# Preditions from the Model.
model.fit(ds_train, epochs = 20, validation_data = ds_test)
loss, accuracy = model.evaluate(ds_test, verbose = 2)
print(f"\nTest-Loss: {loss}, Test-Accuracy: {accuracy}\n\n")
model.summary()
