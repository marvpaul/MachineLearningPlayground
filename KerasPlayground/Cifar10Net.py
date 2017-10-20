from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import np_utils
import numpy
import h5py
import pickle
from PIL import Image


#load a certain number of images from cifar10
def getImages(numberOfTestImages):
    with open('cifar10', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    collectedImages = []
    collectedImageClasses = []
    for value in dict:

        if "data" in str(value, "utf-8"):
            imageNumber = 0
            for pic in dict[value]:
                imageClass = dict[str.encode("labels")][imageNumber]
                collectedImages.append(pic)
                collectedImageClasses.append(imageClass)
                imageNumber += 1
                print('loaded image')
                if imageNumber == numberOfTestImages:
                    break;
    return [collectedImages, [collectedImageClasses]]


#Print a certain pic which is represented as 3072 x 1 vector with RGB values for each pixel
def printImage(pic, name):
    red = pic[0: 1024]
    green = pic[1024: 1024+1024]
    blue = pic[1024+1024: 1024+1024+1024]
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
def createKerasModel():
    model = Sequential()
    model.add(Dense(output_dim= 500, input_dim=3072, activation='relu'))
    model.add(Dense(output_dim= 300, input_dim=500, activation='relu'))
    model.add(Dense(output_dim=10, input_dim=300, activation='softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('cifar10Model.keras')

#Just another try to create a net
def createOtherModel():
    # create model
    model = Sequential([
        Dense(output_dim=3072, input_dim=3072, activation='relu'),
        Dense(output_dim=10, input_dim=3072, activation='softmax'),
    ])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('cifar10Model.keras')


#Create some trainingsdata for the model
def createTrainingData(amountOfData):
    images = getImages(amountOfData)
    inputData = numpy.array(images[0])
    output = []
    for imageClass in images[1][0]:
        classList = numpy.zeros(10)
        classList[imageClass] = 1
        output.append(classList)
    return numpy.array(inputData), output

#Come on, train this shit!
def trainKerasModel(input, output):

    model = load_model('cifar10Model.keras')

    # Fit the model
    model.fit(numpy.array(input), numpy.array(output), epochs=50, batch_size=3)

    scores = model.evaluate(numpy.array(input), numpy.array(output))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save('cifar10Model.keras')

#The accuracy
def testModel(input, output):
    model = load_model('cifar10Model.keras')
    scores = model.evaluate(numpy.array(input), numpy.array(output))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

inputData, outputData = createTrainingData(2000)

inputTrainingsData = inputData[0:1000]
outputTrainingsData = outputData[0:1000]

inputTestData = inputData[1001:1999]
outputTestData = outputData[1001:1999]

#createKerasModel()
#trainKerasModel(inputData, outputData)
testModel(inputTestData, outputTestData)