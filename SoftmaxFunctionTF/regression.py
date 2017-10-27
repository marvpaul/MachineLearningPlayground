# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow


learning_rate = 0.1

# class 0:
# covariance matrix and mean
cov0 = np.array([[5,-4],[-4,4]])
mean0 = np.array([2.,3])
# number of data points
m0 = 100

# class 1
# covariance matrix
cov1 = np.array([[5,-3],[-3,3]])
mean1 = np.array([1.,1])
m1 = 100





# generate m0 gaussian distributed data points with
# mean0 and cov0.
r0 = np.random.multivariate_normal(mean0, cov0, m0)
r1 = np.random.multivariate_normal(mean1, cov1, m1)

# Set model weights
#Input, output
W = tensorflow.Variable(tensorflow.zeros([100, 100]))
b = tensorflow.Variable(tensorflow.zeros([100]))

x1 = tensorflow.placeholder(tensorflow.float32, [None, 100])
y = tensorflow.placeholder(tensorflow.float32, [None, 100])

model = tensorflow.nn.softmax(tensorflow.matmul(x1, W) + b) # Softmax


# Minimize error using cross entropy
cost = tensorflow.reduce_mean((-np.sum(y*tensorflow.log(model)) + np.sum((y - model)*(y - model))), reduction_indices=1)
# Gradient Descent
#optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate)

# Initialize the variables (i.e. assign their default value)
init = tensorflow.global_variables_initializer()


def plot_data(r0, r1):
    plt.figure(figsize=(7.,7.))
    plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
    plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1")
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()

plot_data(r0,r1)