# imports
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def helloBarchart():
        grades = [83,95,91,87,70,6,85,82,100,67,73,77,0]
        decile = lambda grade: grade // 10 * 10
        # cluster grades into deciles and count them
        histogram = Counter(decile(grade) for grade in grades)

        # plot histogram
        plt.bar([x -4 for x in histogram.keys()], # shift each bar to the left by 4
                histogram.values(),                # give each bar its correct height
                8)                                 # give each bar a width of 8
        plt.axis([-5, 105, 0, 5])                  # x-axis from -5 to 105, y-axis from 0 to 5
        plt.xticks([10 * i for i in range(11)])    # x-axis labels at 0, 10, ..., 100
        plt.xlabel("Points rounded")
        plt.ylabel("# of Students")
        plt.title("Distribution of Test Grades")
        plt.show()

helloBarchart()

