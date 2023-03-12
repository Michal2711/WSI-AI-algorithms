from gradient_descent_algorithm import gradient_descent
from plot2D import Plot
from parameters import parameters_g
# import numpy as np
from math import cos, pi, sin


def main():
    for x0, step in parameters_g:
        minimum, parameters = gradient_descent(gradient, x0, step)
        print(f"x0 = {x0}, step = {step} minimum: {minimum}")
        max_abs_x = max(parameters, key=lambda x: abs(x))
        limits_x = {"negative": int(-2*max_abs_x), "positive": int(2*max_abs_x)}  # plot limits of x
        values = [g(x) for x in parameters]
        plot = Plot(parameters, values)
        plot.draw_function(g, limits_x, int(1 / step))
        plot.show()


def g(x):
    return x**2-10*cos(2*pi*x)+10


def gradient(x):
    return 2*x+20*pi*sin(2*pi*x)


if __name__ == "__main__":
    main()
