#Copied from: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/#
import numpy as np
import pandas as pd
import os
import PIL.Image as Image

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D, Conv2D
import matplotlib.pyplot as plt

def getData(number_img, csv_path):
    data = pd.read_csv(csv_path)

    image_paths = data['filename'][:number_img]
    classes = data['label'][:number_img]
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
    return images, classes_vecs

def create_model():
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

images, classes = getData(1200, "data/train.csv")




if os.path.isfile("model.keras"):
    model = load_model("model.keras")
else:
    model = create_model()
    # Fit the model
    model.fit(images, classes, epochs=100, batch_size=100)
    model.save("model.keras")

train_images = images[1000:1040]
train_labels = classes[1000:1040]
predictions = model.predict(train_images)

numbers = []
for pred in predictions:
    numbers.append(np.argmax(pred))
print(numbers)


def plotSampleImages(images, labels):
    image_data_sorted = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(labels)):
        label_nr = np.argmax(labels[i])
        image_data_sorted[label_nr].append(images[i])
    fig, axes = plt.subplots(nrows=15, ncols=10)
    for image_class in range(10):
        axes[0, image_class].set_title(image_class)
        for image in range(len(image_data_sorted[image_class])):
            axes[image, image_class].axis('off')
            axes[image, image_class].imshow((image_data_sorted[image_class][image])/255)

    plt.show()

plotSampleImages(train_images, train_labels)



