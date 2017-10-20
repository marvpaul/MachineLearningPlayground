from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy
import h5py
import pickle
from PIL import Image

modelPath = './cifar/data/cifar10Model.keras'

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
        Dense(output_dim=200, input_dim=3072, activation='relu'),
        Dense(output_dim=200, input_dim=200, activation='relu'),
        Dense(output_dim=10, input_dim=200, activation='softmax'),
    ])
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save(modelPath)


# Just another try to create a net
def createOtherModel():
    # create model
    model = Sequential([
        Dense(output_dim=3072, input_dim=3072, activation='relu'),
        Dense(output_dim=10, input_dim=3072, activation='softmax'),
    ])
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.save('cifar10Model.keras')


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
    model.fit(numpy.array(input), numpy.array(output), epochs=50, batch_size=100)

    scores = model.evaluate(numpy.array(input), numpy.array(output))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    model.save(modelPath)


# Test accuracy
def testModel(input, output):
    model = load_model(modelPath)
    scores = model.evaluate(numpy.array(input), numpy.array(output))
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


inputData, outputData = createTrainingData()
print('Loaded ' + str(len(inputData)) + 'images')

inputTrainingsData = inputData[0:48999]
outputTrainingsData = outputData[0:48999]

inputTestData = inputData[49000:49999]
outputTestData = outputData[49000:49999]


#createKerasModel(inputTrainingsData, outputTrainingsData)
#trainKerasModel(inputTrainingsData, outputTrainingsData)
testModel(inputTestData, outputTestData)
