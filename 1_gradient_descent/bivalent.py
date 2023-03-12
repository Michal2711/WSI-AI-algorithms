from gradient_descent_algorithm import gradient_descent
from plot3D import Plot3D
from parameters import parameters_f
import numpy as np
from math import ceil, floor


def main():
    for start_point, step in parameters_f:
        minimum, points = gradient_descent(gradient, start_point, step)
        print(f"Starting point: {start_point}, step = {step}) minimum: {minimum}")

        X, Y, Z = setting_XYZ(points)

        x_min, y_min, z_min = floor(2*min(X)), floor(2*min(Y)), floor(2*min(Z))
        x_max, y_max, z_max = ceil(2*max(X)), ceil(2*max(Y)), ceil(2*max(Z))
        limits = {"x": [x_min, x_max], "y": [y_min, y_max], "z": [z_min, z_max]}

        plot = Plot3D(X, Y, Z, limits)
        plot.draw_function(f)
        plot.show()


def setting_XYZ(points):
    X, Y = list(), list()
    for x, y in points:
        X.append(x)
        Y.append(y)
    Z = [f(np.array([x, y])) for x, y in zip(X, Y)]
    return X, Y, Z


def f(x):
    return x[0]**2+x[1]**2


def gradient(x):
    return 2*x[0], 2*x[1]


if __name__ == "__main__":
    main()
