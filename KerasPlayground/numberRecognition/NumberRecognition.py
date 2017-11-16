#Datasource:http://yann.lecun.com/exdb/mnist/
import numpy as np
import pandas as pd
import os
import PIL.Image as Image

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D
import matplotlib.pyplot as plt

def getData(number_img, csv_path):
    '''
    Read images, and save to array
    :param number_img: the number of images to load
    :param csv_path: the path for CSV in which the image paths are stored
    :return: image_array, classes_array
    '''
    data = pd.read_csv(csv_path)

    image_paths = data['filename'][:number_img]

    #Create class vectors with following shape: [0,0,0,0,0,0,0,0,0,1] --> Represents class for number 9
    classes = data['label'][:number_img]
    classes_vecs = np.array([np.zeros(10) for classes in classes])
    for class_nr in range(len(classes)):
        classes_vecs[class_nr][classes[class_nr]] = 1

    images = []
    for i in range(len(image_paths)):
        path = os.path.join('data/Images/train', image_paths[i])
        img = Image.open(path).convert('RGBA')
        arr = np.array(img)
        images.append(arr)

    images = np.array(images)
    return images, classes_vecs

def create_model():
    '''Create a keras model with some conv layers'''
    modelK = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=images[0].shape),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    modelK.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return modelK

#Read and process data
images, classes = getData(1500, "data/train.csv")



#Check if keras model exists. In case it exists, just run prediction, otherwise create, train and save a new model
if os.path.isfile("model.keras"):
    model = load_model("model.keras")
else:
    model = create_model()

    # Fit the model
    model.fit(images, classes, epochs=50, batch_size=100)
    model.save("model.keras")

test_images = images[1000:1060]
#Lets make some predictions :)
predictions = model.predict(np.array(test_images))

numbers = []
for pred in predictions:
    numbers.append(np.argmax(pred))


def plotSampleImages(images, labels):
    '''
    Plot all predicted numbers
    :param images: the images to plot
    :param labels: the labels which were predicted
    :return: nothing :O
    '''
    image_data_sorted = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(labels)):
        image_data_sorted[labels[i]].append(images[i])
    fig, axes = plt.subplots(nrows=15, ncols=10)
    for image_class in range(10):
        axes[0, image_class].set_title(image_class)
        for image in range(len(image_data_sorted[image_class])):
            axes[image, image_class].axis('off')
            axes[image, image_class].imshow((image_data_sorted[image_class][image])/255)

    plt.show()

plotSampleImages(test_images, numbers)



'''
Code for parsing some self drawn images :) 
test_images = []
img = Image.open('data/Images/own/Unbenannt-1.png').convert('RGBA')
arr = np.array(img)
test_images.append(arr)

img = Image.open('data/Images/own/Unbenannt-2.png').convert('RGBA')
arr = np.array(img)
test_images.append(arr)

img = Image.open('data/Images/own/Unbenannt-3.png').convert('RGBA')
arr = np.array(img)
test_images.append(arr)

img = Image.open('data/Images/own/Unbenannt-4.png').convert('RGBA')
arr = np.array(img)
test_images.append(arr)
'''