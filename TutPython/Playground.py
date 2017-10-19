# %matplotlib inline you need that command to plot inside the notebook
# matplotlib inline

# imports
import matplotlib.pyplot as plt
import numpy as np

def drawSinCos():
    plt.style.use('fivethirtyeight')
    # single line plot
    # set values for x and y coordinates
    x = np.linspace(0, 2 * np.pi, 400)
    y1 = np.sin(x)
    y2 = np.cos(x)
    # create axes and plot two functions
    plt.plot(x, y1, x, y2)

def drawSomeWithLabels():
    fig = plt.figure()
    x = np.linspace(0, 2 * np.pi, 400)
    # plot the functions
    x1, = plt.plot(x, x, label=r"$x$")
    x2, = plt.plot(x, x**2, label=r"$x^2$")
    x3, = plt.plot(x, x**3, label=r"$x^3$")
    # set annotations
    plt.xlabel(r'$x$', fontsize=12)
    plt.ylabel(r'$y$', fontsize=12)
    plt.title('two function plot');
    plt.legend(loc=2)

def task1():
    fig = plt.figure()
    x = range(0,9)
    variance = np.array([1,2,4,8,16,32,64,128,256])
    bias_squared = np.array(variance[::-1])
    plt.plot(x, variance, color='green', label='Variance')
    plt.plot(x, bias_squared, 'red', ls='-.', lw=0.9, label='Bias')
    plt.plot(x, (variance + bias_squared), 'blue', ls=':', label='Total error')
    plt.title('Model complexity')
    plt.legend(loc=9)
task1()
plt.show()


# prints available out-of-the-box styles
print(plt.style.available)
# choose a diffrent style and plot an example

#fig = plt.figure()
#ax = plt.axes()

