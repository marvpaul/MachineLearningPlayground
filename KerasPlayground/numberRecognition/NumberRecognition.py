#Copied from: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/#
import numpy as np
import pandas as pd
import os
import PIL.Image as Image

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Conv2D

data = pd.read_csv("data/train.csv")

image_paths = data['filename'][:1000]
classes = data['label'][:1000]
classes_vecs = []
for class_nr in classes:
    class_vec = np.zeros(10)
    class_vec[class_nr] = 1
    classes_vecs.append(class_vec)
classes_vecs = np.array(classes_vecs)
images = []

for i in range(len(image_paths)):
    path = os.path.join('data/Images/train', image_paths[i])
    img = Image.open(path).convert('RGBA')
    arr = np.array(img)
    images.append(arr)

images = np.array(images)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=images[0].shape),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(images, classes_vecs, epochs=100, batch_size=100)

