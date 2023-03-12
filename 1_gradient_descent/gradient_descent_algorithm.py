from typing import Tuple
import numpy as np


def gradient_descent(gradient, x0, step) -> Tuple[float, list]:
    Iterations = 850
    x = np.array(x0, dtype=np.float)
    list_of_x = [np.copy(x)]  # list of current x
    while Iterations > 0:
        x -= calculation(gradient, x, step)
        list_of_x.append(np.copy(x))
        Iterations -= 1
    return x, list_of_x


def calculation(gradient, x, step):
    difference = step * np.array(gradient(x), dtype=np.float)  # dla f wielowartosciowych * nie dziala tak jakbysmy chcieli wiec trzeba np.array
    return difference
