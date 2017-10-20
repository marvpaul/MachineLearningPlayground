#from keras.models import Sequential
#from keras.layers import Dense
#from keras.models import load_model
import numpy
import h5py

#Create model, train and save
def createModelInitial():
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load pima indians dataset
    dataset = numpy.loadtxt("diabeticData.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    # Fit the model
    model.fit(X, Y, epochs=150, batch_size=10)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save('model.keras')

def loadAndContinueTraining():
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load pima indians dataset
    dataset = numpy.loadtxt("diabeticData.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    model = load_model('model.keras')
    model.fit(X, Y, epochs=1000, batch_size=10)

    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    model.save('model.keras')

def makePredictions():
    # calculate predictions
    predictions = model.predict(X)
    # round predictions
    rounded = [round(x[0]) for x in predictions]
    print(rounded)
    print("sth")

dataset = numpy.loadtxt("diabeticData.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

print(X)