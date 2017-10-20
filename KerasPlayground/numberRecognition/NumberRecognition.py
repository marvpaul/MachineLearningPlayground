#Copied from: https://www.analyticsvidhya.com/blog/2016/10/tutorial-optimizing-neural-networks-using-keras-with-image-recognition-case-study/#
import numpy as np
import pandas as pd
import os

from keras.models import load_model
import pylab
from scipy.misc import imread
from sklearn.metrics import accuracy_score

import tensorflow as tf
import keras



# To stop potential randomness
seed = 128
rng = np.random.RandomState(seed)


train = pd.read_csv(os.path.join('data', 'train.csv'))
test = pd.read_csv(os.path.join('data', 'Test.csv'))

sample_submission = pd.read_csv(os.path.join('data', 'sample_submission.csv'))

train.head()

img_name = rng.choice(train.filename)
filepath = os.path.join('data', 'Images', 'train', img_name)

img = imread(filepath, flatten=True)

temp = []
for img_name in train.filename:
    image_path = os.path.join('data', 'Images', 'train', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

train_x = np.stack(temp)

train_x /= 255.0
train_x = train_x.reshape(-1, 784).astype('float32')
'''
temp = []
for img_name in test.filename:
    image_path = os.path.join('data', 'Images', 'test', img_name)
    img = imread(image_path, flatten=True)
    img = img.astype('float32')
    temp.append(img)

test_x = np.stack(temp)

test_x /= 255.0
test_x = test_x.reshape(-1, 784).astype('float32')
'''

train_y = keras.utils.np_utils.to_categorical(train.label.values)

split_size = int(train_x.shape[0]*0.7)

train_x, val_x = train_x[:split_size], train_x[split_size:]
train_y, val_y = train_y[:split_size], train_y[split_size:]

train.label.ix[split_size:]


# define vars
input_num_units = 784
hidden_num_units = 50
output_num_units = 10

epochs = 5
batch_size = 128

# import keras modules

from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential([
    Dense(output_dim=hidden_num_units, input_dim=input_num_units, activation='relu'),
    Dense(output_dim=output_num_units, input_dim=hidden_num_units, activation='softmax'),
])

# compile the model with necessary attributes
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

trained_model = model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, validation_data=(val_x, val_y))

model.save('./data/model')