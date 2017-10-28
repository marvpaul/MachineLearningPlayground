# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow


learning_rate = 1

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


# Parameters
learning_rate = 1
training_epochs = 25
batch_size = 5
display_step = 1



# generate m0 gaussian distributed data points with
# mean0 and cov0.
r0 = np.random.multivariate_normal(mean0, cov0, m0)
r1 = np.random.multivariate_normal(mean1, cov1, m1)

inputData = np.transpose(np.array([(r0[...,0][:95]) + (r1[...,0][:95]), (r0[...,1][:95] + (r1[...,1][:95]))]))

out_data = np.append(np.ones(95), np.zeros(95))

#out_trainingsdata = (r0[...,1][95:])(r1[...,1][95:])

# Set model weights
#Input, output
W = tensorflow.Variable(tensorflow.zeros([2, 1]))
b = tensorflow.Variable(tensorflow.zeros([1]))

x = tensorflow.placeholder(tensorflow.float32, [None, 2])
y = tensorflow.placeholder(tensorflow.float32, [None, 1])

model = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b) # Softmax, logistic regression

# Minimize error using cross entropy without using LD2 log
cost = tensorflow.reduce_mean(-tensorflow.reduce_sum(y*tensorflow.log(model), reduction_indices=1))

# Gradient Descent
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tensorflow.global_variables_initializer()

with tensorflow.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(inputData)/batch_size)
        # Loop over all batches
        for i in range(len(inputData)):
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: [inputData[i]],
                                                          y: [[out_data[i]]]})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tensorflow.equal(tensorflow.argmax(model, 1), tensorflow.argmax(y, 1))
    print(correct_prediction)
    # Calculate accuracy for 3000 examples
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
    #print("Accuracy:", accuracy.eval({x: [inputData], y: [out_data]}))



def plot_data(r0, r1):
    plt.figure(figsize=(7.,7.))
    plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
    plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1")
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()



plot_data(r0,r1)