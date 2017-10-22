from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D
import numpy
import h5py
import pickle
from PIL import Image
import operator

modelPath = './cifar/data/cifar10Model3.keras'
imagesClasses = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# load images from cifar10
def getImages():
    dics = []
    paths = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    for path in paths:
        with open('./cifar/data/' + path, 'rb') as fo:
            dics.append(pickle.load(fo, encoding='bytes'))

    collected_images = []
    collected_image_classes = []
    for dict in dics:
        for value in dict:
            if "data" in str(value, "utf-8"):
                image_number = 0
                for pic in dict[value]:
                    imageClass = dict[str.encode("labels")][image_number]
                    collected_images.append(pic)
                    collected_image_classes.append(imageClass)
                    image_number += 1
    return [collected_images, [collected_image_classes]]


# Print a certain pic which is represented as 3072 x 1 vector with RGB values for each pixel
def printImage(pic, name):
    red = pic[0: 1024]
    green = pic[1024: 1024 + 1024]
    blue = pic[1024 + 1024: 1024 + 1024 + 1024]
    w, h = 32, 32
    data = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    pixelNum = 0
    for row in range(0, len(data)):
        for pixel in range(0, len(data[row])):
            data[row, pixel] = [red[pixelNum], green[pixelNum], blue[pixelNum]]
            pixelNum += 1

    img = Image.fromarray(data, 'RGB')
    img.save(str(name) + '.png')
    img.show()


'''Create a keras model with input of 3072 and output of 10
    which represents the number of classes cifar10 has defined'''
def createKerasModel(input, output):
    # create model
    model = Sequential([
        Dense(output_dim=250, input_dim=3072, activation='relu'),
        Dense(output_dim=100, input_dim=250, activation='relu'),
        Dense(output_dim=50, input_dim=100, activation='relu'),
        Dense(output_dim=25, input_dim=50, activation='relu'),
        Dense(output_dim=10, input_dim=25, activation='softmax'),
    ])
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(modelPath)


#using the model mentioned here: https://elitedatascience.com/keras-tutorial-deep-learning-in-python
def createKerasModel3():
    #create model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(32,32,3)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.save(modelPath)


# Create some trainingsdata for the model
def createTrainingData():
    images = getImages()
    inputData = numpy.array(images[0])
    output = []
    for imageClass in images[1][0]:
        classList = numpy.zeros(10)
        classList[imageClass] = 1
        output.append(classList)
    return numpy.array(inputData) / 255, output


# Come on, train this shit!
def trainKerasModel(input, output):
    model = load_model(modelPath)

    # Fit the model
    model.fit(input, output, epochs=100, batch_size=1000)

    scores = model.evaluate(input, output)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    model.save(modelPath)


# Test accuracy
def testModel(input, output):
    model = load_model(modelPath)
    scores = model.evaluate(numpy.array(input), numpy.array(output))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def makePrediction(image):
    model = load_model(modelPath)
    prediction = model.predict(image)
    index, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))
    print(prediction)
    print('The net think your images is a:' + imagesClasses[index])

#Just reshape the array format of given images to fit to neural net structure
def prepareInputImages(images):
    images = numpy.array(images)
    return images.reshape(len(images), 32, 32, 3)


inputData, outputData = createTrainingData()
print('Loaded ' + str(len(inputData)) + 'images')

inputTrainingsData = prepareInputImages(inputData[:49000])
outputTrainingsData = numpy.array(outputData[:49000])

inputTestData = prepareInputImages(inputData[49000:49999])
outputTestData = numpy.array(outputData[49000:49999])

predictImage = numpy.array(inputData[49887])
printImage(predictImage*255, 'test')
predictImage = predictImage.reshape(1, 32, 32, 3)

makePrediction(predictImage)


#createKerasModel3()
#trainKerasModel(inputTrainingsData, outputTrainingsData)
#testModel(inputTestData, outputTestData)
