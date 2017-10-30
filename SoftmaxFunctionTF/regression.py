# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow



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

# Parameters for model training
learning_rate = 0.1
training_epochs = 100
batch_size = 20
display_step = 1

#Transform the data for model training :)
inputData = np.array(np.transpose([np.append(r0[...,0][:100], r1[...,0][:100]), np.append(r0[...,1][:100], r1[...,1][:100])]), dtype="float32")
out_data = np.array(np.transpose([np.append(np.ones(100), np.zeros(100)), np.append(np.zeros(100), np.ones(100))]), dtype="float32")

#Shuffle data
data = []
for i in range(len(inputData)):
    data.append(np.append(inputData[i], out_data[i]))
np.random.shuffle(data)
inputData = []
out_data = []
for i in range(len(data)):
    inputData.append(data[i][:2])
    out_data.append(data[i][2:])

#Define some placeholders and variables
y = tensorflow.placeholder(tensorflow.float32, [None, 2])
x = tensorflow.placeholder(tensorflow.float32, [None, 2])
W = tensorflow.Variable(tensorflow.zeros([2, 2]))
b = tensorflow.Variable(tensorflow.zeros([1]))

#The model
model = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)
#model = tensorflow.nn.softmax(tensorflow.add((tensorflow.matmul(x1, W1) + b), tensorflow.matmul(x2, W2) + b)) # Softmax, logistic regression

# Minimize error using cross entropy with l2 regularization
l2 = tensorflow.reduce_sum(tensorflow.pow(y-model, 2))
cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(logits=tensorflow.matmul(x, W) + b, labels=y)
cost = cross_entropy  + l2
#cross_entropy = tensorflow.reduce_mean(- tensorflow.log(model)) + tensorflow.reduce_sum(- tensorflow.log(1 - model))
#cross_entropy = - tensorflow.reduce_mean(y * tensorflow.log(model + (1-y)*tensorflow.log(1-model)))

# Using gradient descent as optimizer
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
        for i in range(total_batch):
            # Fit training using batch data
            x_batch = inputData[batch_size*i:batch_size*(i+1)]
            y_batch = out_data[batch_size*i:batch_size*(i+1)]
            _, c = sess.run([optimizer, cost], feed_dict={x: x_batch, y: y_batch})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            #print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            print("Epoch", epoch+1, avg_cost[0])

    print("Optimization Finished!")

    # Test model
    correct_prediction = tensorflow.equal(tensorflow.argmax(model, 1), tensorflow.argmax(y, 1))
    print(correct_prediction)
    # Calculate accuracy for the trainings data
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
    print("Accuracy:", accuracy.eval({x: inputData, y: out_data}))

    #Classify all trainings data and plot
    classification = sess.run(model, feed_dict={x: inputData})
    class1 = []
    class2 = []
    for pointNumber in range(len(classification)):
        if classification[pointNumber][0] > classification[pointNumber][1]:
            class1.append(inputData[pointNumber])
        else:
            class2.append(inputData[pointNumber])


def plot_data(r0, r1):
    plt.figure(figsize=(7.,7.))
    plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
    plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1")
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")
    plt.show()

plot_data(np.array(class1),np.array(class2))
plot_data(r0, r1)
