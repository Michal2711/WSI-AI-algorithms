import matplotlib.pyplot as plt
import numpy as np


class Plot3D:
    def __init__(self, x, y, z, limits):
        self.limits = limits
        self.setting_ax()
        self.showing_points(x, y, z)

    def show(self):
        plt.show()

    def setting_ax(self):
        self.ax = plt.axes(projection='3d')  # add an axes to the current figure and make it the current axes

        self.ax.set_xlim(self.limits["x"][0], self.limits["x"][1])  # Set the x-axis view limits.
        self.ax.set_ylim(self.limits["y"][0], self.limits["y"][1])  # Set the y-axis view limits.
        self.ax.set_zlim(self.limits["z"][0], self.limits["z"][1])  # Set the z-axis view limits.

    def showing_points(self, x, y, z):
        self.ax.scatter(x[:-1], y[:-1], z[:-1], color="y", alpha=1)  # alpha = 1 -> opaque  alpha = 0 -> transparent
        self.ax.scatter(x[-1], y[-1], z[-1], color="g", alpha=1)

    def draw_function(self, function):
        X = np.array([x for x in range(self.limits["x"][0], self.limits["x"][1])])  # has to be array not list ( attribute ndim )
        Y = np.array([y for y in range(self.limits["y"][0], self.limits["y"][1])])
        X, Y = np.meshgrid(X, Y)  # meshgrid return coordinate matrices from coordinate vectors.
        Z = np.array([function([x, y]) for x, y in zip(X, Y)])
        self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, antialiased=True,
                             cmap='viridis')  # antialiased = False -> surface is opaque
                                              # when rstride and cstride are not 1, not all points are used to draw the surface
