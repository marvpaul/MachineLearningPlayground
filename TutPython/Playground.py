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

#Draw some lines
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

plt.show()

def multiplePlots():
    # some function
    x = np.linspace(0, 1,25 * np.pi, 400)
    y = x ** 2

    # create an container
    fig = plt.figure()

    # add two grids with specific positopn [left, bottom, width, height] to the container
    main_axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
    inserted_axes = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # inset axes
    inserted_axes2 = fig.add_axes([0.6, 0.6, 1, 1])

    # plot main figure
    main_axes.plot(x, y, 'b')
    main_axes.set_title('main plot')

    # plot insert figure
    inserted_axes.plot(y, x, 'y')
    inserted_axes.set_title('inserted plot')

    inserted_axes2.plot(x, y, 'c')
    inserted_axes2.set_title('Sth')
    plt.show()

def subplots():
    # some function
    x = np.linspace(0, 2 * np.pi, 400)
    y = x

    # create a figure with specific rows and colums
    fig, axes = plt.subplots(nrows=3, ncols=3)

    # plot grids and define position via [row,colum]
    axes[0,0].plot(x, y + 2)
    axes[0,1].plot(x, y % 2)
    axes[0,2].plot(x, y % 2)
    axes[1,0].plot(x, y ** 2)
    axes[1,1].plot(x, y / 2)
    axes[1,2].plot(x, y / 2)
    axes[2,0].plot(x, y / 2)
    axes[2,1].plot(x, y / 2)
    axes[2,2].plot(x, y / 2)
    plt.show()

def twinAxes():
    # some function
    x = np.linspace(0, 2 * np.pi, 400)
    y = x
    # twin axes
    fig, ax = plt.subplots()

    # plot x^2 in red
    ax.plot(x, x**2, 'r-.')
    ax.set_ylabel(r"area $m^2$", color='r')
    for label in ax.get_yticklabels():
        label.set_color('r')

    # plot x^3 in the same grid with an additional axes
    ax_twin = ax.twinx()
    ax_twin.plot(x, x**3, 'b--')
    ax_twin.set_ylabel(r"volume $m^3$", color="b")
    for label in ax_twin.get_yticklabels():
        label.set_color('b')
    plt.show()

def axes_manipulation():
    # some function
    x = np.linspace(0.1, 200 * np.pi, 40000)
    y = x
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    # default matplotlib plot
    ax[0].plot(x, x**2, x, x**3)
    ax[0].set_title("default range")

    # set axes by option: ‘on’, ‘off’, ‘equal’, ‘tight’, ‘scaled’, ‘auto’, ‘image’, ‘square’
    ax[1].plot(x, x**2, x, x**3)
    ax[1].axis('tight')
    ax[1].set_title("option: tight ")

    # customize axes with individual values [min x, max x, min y, max y]
    ax[2].plot(x, x**2, x, x**3)
    ax[2].axis([1, 20,0,10000])
    ax[2].set_title("custom range");
    # add annotations
    ax[0].text(5, 50, r"$y=x^2$", fontsize=20, color="b")
    ax[1].text(5.5, 150, r"$y=x^3$", fontsize=20, color="r");
    plt.show()


def scatterPlot():
    # model data
    network_friends = [ 70, 65, 72, 63, 71, 64, 60, 64, 67]
    stayed_minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
    person = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    # scatterplot
    plt.scatter(network_friends, stayed_minutes)

    # label each point
    for person, friend_count, minute_count in zip(person, network_friends, stayed_minutes):
        plt.annotate(person, xy=(friend_count, minute_count),
                     xytext=(5, -5), # offset from each point
                     textcoords='offset points') # set offset
        plt.title("Daily Minutes vs. Number of Friends")
        plt.xlabel("# friends in the network")
        plt.ylabel("minutes spent on the network")
    plt.show()

def task2():
    plt.style.use('seaborn')
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # simple test grade plot
    test_1_grades = [ 99, 90, 85, 97, 80]
    test_2_grades = [100, 85, 60, 90, 70]
    labels = ['Noam', 'Marie', 'Kurt', 'Alexandra', 'Alexander']

    axes[0].scatter(test_1_grades, test_2_grades)
    axes[0].axis([50, 100,50,100])

    axes[1].scatter(test_1_grades, test_2_grades)
    axes[1].axis([50,100,50,100])

    for i, text in enumerate(labels):
        axes[1].annotate(text, (test_1_grades[i], test_2_grades[i]))

    plt.title("Plot with comparable axis")
    plt.ylabel('test 2 grades')
    plt.xlabel('test 1 grades')
    #plt.scatter(test_1_grades, test_2_grades)
    plt.show()


task2()

# prints available out-of-the-box styles
print(plt.style.available)
# choose a diffrent style and plot an example

#fig = plt.figure()
#ax = plt.axes()

