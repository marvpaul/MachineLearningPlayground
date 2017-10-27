# Initialization of some basic modules and global parameters

# imports
import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from KNearestNeighbor import KNearestNeighbor



from scipy._lib.six import xrange

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.show()

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#%load_ext autoreload
#%autoreload 2

def load_CIFAR_batch(filename):
    """
    Opens and decodes a single batch of the cifar10 dataset.

    Input:
    - A string containing the absolute path to a batch file

    Output:
    - An array containing the data
    - An array containing the labels
    """
    with open(filename, 'rb') as f:
        #####################################################################
        # TODO (2):                                                         #
        # Use pickle.load() to read the cifar10 data into a dictonary.      #
        # Create a parameter X for the image data and a parameter Y for the #
        # labels.                                                           #
        #####################################################################
        datadict = pickle.load(f, encoding='bytes')
        X = []
        Y = []
        for value in datadict:
            if "data" in str(value, "utf-8"):
                image_number = 0
                for pic in datadict[value]:
                    imageClass = datadict[str.encode("labels")][image_number]
                    X.append(pic)
                    Y.append(imageClass)
                    image_number += 1
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        num_images = 10000
        X = np.array(X)
        X = X.reshape(num_images, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(root_folder):
    """
    Load training and test data of the cifar10 dataset into arrays.

    Input:
    - A string containing the absolute path to the cifar 10 dataset folder

    Output:
    - Array containing the image data for the trainings set
    - Array containing the image labels for the trainings set
    - Array containing the image data for the test set
    - Array containing the image labels for the test set
    """
    xs = []
    ys = []
    # Load and decode each batch of the trainings set
    #for batch in range(1,6):
    for batch in range(1,2):
        f = os.path.join(root_folder, 'data_batch_%d' % (batch, ))
        print(f)
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
        # Create one ndarray from the single batches for data and labels
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    # Load and decode test data and labels
    Xte, Yte = load_CIFAR_batch(os.path.join(root_folder, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

#####################################################################
# TODO (0):                                                         #
# Download the Cifar10 Python version into your directroy, unzip    #
# and point to it.                                                  #
#####################################################################
cifar10_dir = 'cifar/data/'  # default dir
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# Plot k random examples of training images from each class.
# Answer:
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
#####################################################################
# TODO (3):                                                         #
# Plot a figure where the colums are equal to the number of classes #
# and rows are defined by the number of samples per class. Each     #
# sample should be a random image from training data. Add class     #
# labels as title and remove axis from the figure.                  #
#                                                                   #
# Hint: Keep an eye on the cifar10 data encoding.                   #
#####################################################################

def plotSampleImages(images, labels):
    #Get a certain number of random sample images for each label
    index = random.randint(0, len(images))
    images_per_class = [[] for x in xrange(10)]
    for image in images:
        if len(images_per_class[labels[index]]) < samples_per_class:
            images_per_class[labels[index]].append(images[index])
        index += 1
        if index == len(images):
            index = 0
    fig, axes = plt.subplots(nrows=samples_per_class, ncols=10)

    #plot all images and add labels
    class_number = 0
    image_number = 0
    for image_class in images_per_class:
        axes[0, class_number].set_title(classes[class_number])
        for image in image_class:
            axes[image_number, class_number].axis('off')
            axes[image_number, class_number].imshow((image)/255)
            image_number += 1
        class_number += 1
        image_number = 0
    plt.show()

#plotSampleImages(X_train, y_train)
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################

# Subsample trainings data
num_training = 5000 #5000
mask = range(num_training)
X_train = X_train[mask]
y_train = y_train[mask]

# Subsample test data
num_test = 50 #500
mask = range(num_test)
X_test = X_test[mask]
y_test = y_test[mask]

# Reshape each image data into a 1-dim array
print (X_train.shape, X_test.shape) # Should be: (5000, 32, 32, 3) (500, 32, 32, 3)

#####################################################################
# TODO (2):                                                         #
# Reshape the image data to one dimension.                          #
#                                                                   #
# Hint: Look at the numpy reshape function and have a look at -1    #
#       option                                                      #
#####################################################################
X_train = np.array(X_train).reshape(-1, 3072)
X_test = np.array(X_test).reshape(-1, 3072)
#####################################################################
#                       END OF YOUR CODE                            #
#####################################################################

print (X_train.shape, X_test.shape) # Should be: (5000, 3072) (500, 3072)

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

# Before running this cell: open `ak1163051/k_nearest_neighbor.py` and 'implement compute_distances_with_loops`

# Compute the distance between the test data and trainings data
dists = classifier.compute_distances_with_loops(X_test)
print(dists.shape) # Should be: (500, 5000)


#dists_vec = classifier.compute_distances_vectorized(X_test)
# This task is not vital for the notebook. Run that cell only if you have implement the vectorized function


# check that the distance matrix agrees with the one we computed before:
'''difference = np.linalg.norm(dists - dists_vec, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')
'''
# Before running this cell: implement the function predict_labels
y_test_pred = classifier.predict_labels(dists,1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO (2):                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
#                                                                              #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, [int(len(X_train)/num_folds*4), len(X_train)-1])
y_train_folds = np.array_split(y_train, [int(len(X_train)/num_folds*4), len(X_train)-1] )
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO (5):                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
X_val = X_train_folds[1]
y_val = y_train_folds[1]
X_train = X_train_folds[0]
Y_train = y_train_folds[0]

for k in k_choices:
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances_with_loops(X_test)
    y_test_pred = classifier.predict_labels(dists,k)
    accuracy = float(np.sum(y_test_pred == y_test)) / len(X_val)
    k_to_accuracies.update({k : accuracy})
    #TODO: Make some more loops :O
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################
for k in sorted(k_to_accuracies):
        print('k = %d, accuracy = %f' % (k, k_to_accuracies))
# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))