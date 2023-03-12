import matplotlib.pyplot as plt
import numpy as np


class Plot:
    def __init__(self, parameters, values):
        self.minimum_point = (parameters[-1], values[-1])
        self.showing_points(parameters, values)
        self.settings()

    def show(self):
        plt.show()

    def showing_points(self, parameters, values):
        plt.scatter(parameters[:-1], values[:-1], color="y")  # getting each element from lists except last and showing points
        plt.scatter(self.minimum_point[0], self.minimum_point[1], color="g")  # getting last element from lists and showing minimum

    def settings(self):
        plot = plt.gca()
        plot.spines['top'].set_color('none')
        plot.spines['bottom'].set_position('zero')
        plot.spines['left'].set_position('zero')
        plot.spines['right'].set_color('none')

    def draw_function(self, function, limits_x, precision):
        x_difference = abs(limits_x["positive"] - limits_x["negative"])
        parameters = np.linspace(limits_x["negative"], limits_x["positive"], num=precision * x_difference)
        values = [function(x) for x in parameters]
        plt.plot(parameters, values)
