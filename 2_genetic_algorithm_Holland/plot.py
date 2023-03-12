import matplotlib.pyplot as plt


class Plot:
    def __init__(self, parameters, values):
        self.minimum_point = (parameters[-1], values[-1])
        self.showing_points(parameters, values)
        self.settings()

    def show(self):
        plt.show()

    def showing_points(self, parameters, values):
        plt.scatter(parameters, values, color="r")

    def settings(self):
        plot = plt.gca()
        plot.spines['top'].set_color('none')
        plot.spines['bottom'].set_position('zero')
        plot.spines['left'].set_position('zero')
        plot.spines['right'].set_color('none')
