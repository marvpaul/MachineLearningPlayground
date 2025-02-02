# imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow

'''
Where r0 is a dataset which represents data from class 1, r1 dataset for class 2
'''
def preparingData(r0, r1):
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
    return inputData, out_data


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

#Preparing data
inputData, out_data = preparingData(r0, r1)

# Parameters for model training
learning_rate = 0.01
training_epochs = 10000
batch_size = 20
display_step = 1

#Define some placeholders and variables
y = tensorflow.placeholder(tensorflow.float32, [None, 2])
x = tensorflow.placeholder(tensorflow.float32, [None, 2])
W = tensorflow.Variable(tensorflow.zeros([2, 2]))
b = tensorflow.Variable(tensorflow.zeros([2]))

#The model
model = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)

# Minimize error using cross entropy with l2 regularization
cross_entropy = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
l2 = 0.01 * tensorflow.nn.l2_loss(W)
cost = cross_entropy + l2

# Using gradient descent as optimizer
optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(cost)


# Create and initialize a session
sess = tensorflow.Session()
init = tensorflow.global_variables_initializer()
sess.run(init)

bias = 0;
weight = 0;
cost_data = [[],[]]
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
            print("Epoch", epoch+1, "Loss: ", avg_cost)
            cost_data[0].append(epoch+1)
            cost_data[1].append(avg_cost)

    print("Optimization Finished!")

    # Calculate accuracy for the trainings data
    accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(model,1), tensorflow.argmax(y,1)), "float"))
    print("Accuracy:", accuracy.eval({x: inputData, y: out_data}))

    weight = sess.run(W)
    bias = sess.run(b)
    print(print("W: ", weight))
    print(print("Bias: ", bias))


def plot_data_and_bounday():
    #Calculate and plot the decision boundary :)
    x_values = [item[0] for item in inputData]
    x_decision_bound = np.linspace(min(x_values), max(x_values))
    y_decision_bound = (-bias[0] - weight[:,0][0]*x_decision_bound) / weight[:,0][1]
    plt.figure(figsize=(7.,7.))
    plt.scatter(r0[...,0], r0[...,1], c='b', marker='o', label="Klasse 0")
    plt.scatter(r1[...,0], r1[...,1], c='r', marker='x', label="Klasse 1")
    plt.plot(x_decision_bound, y_decision_bound)
    plt.xlabel("$x_0$")
    plt.ylabel("$x_1$")

def plot_costs_per_iteration():
    #Plot costs per iteration
    plt.figure(2)
    plt.subplot(111)
    plt.plot(cost_data[0], cost_data[1])
    plt.xlabel("# of iterations / epochs")
    plt.ylabel("costs")
    plt.title("Learning progress")

plot_data_and_bounday()

plot_costs_per_iteration()

plt.show()


'''
    for i in range(len(classification)):
        classification[i][0] = int(round(classification[i][0]))
        classification[i][1] = int(round(classification[i][1]))
    print("Break")
    counter = 0
    for i in range(len(classification)):
        if (classification[i]==out_data[i]).all():
            counter += 1
    print(counter)
    
    
    #Classify all trainings data and plot
    classification = sess.run(model, feed_dict={x: inputData})
    class1 = []
    class2 = []
    for pointNumber in range(len(classification)):
        if classification[pointNumber][0] > classification[pointNumber][1]:
            class1.append(inputData[pointNumber])
        else:
            class2.append(inputData[pointNumber])
'''