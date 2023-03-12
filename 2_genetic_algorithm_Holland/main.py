from Holland import Holland
from individuals import generate_population, precise_Individuals
from math import sin, sqrt
import numpy as np
from plot import Plot


def price_q(x):
    x1, x2, x3, x4 = x
    return (x1 + 2*x2 - 7)**2 + \
        (2*x1 + x2 - 5)**2 + \
        sin(1.5*x3)**3 + \
        ((x3-1)**2)*(1+sin(1.5*x4)**2) + \
        ((x4 - 1)**2)*(1 + sin(x4)**2)


def wariancja(Q, avg):
    sigma = 0.0
    for q in Q:
        sigma += (q - avg)**2
    return sigma/len(Q)


def main():
    P_0 = generate_population(30)
    # P_0 = precise_Individuals
    for i in range(len(P_0)):
        print(f'i: {i} - {P_0[i]}')
    RESULT = list()
    for i in range(25):
        best, q_result = Holland(price_q, P_0, len(P_0), 0.05, 0.4, 1000)
        RESULT.append(q_result)
    # print(best)
    for i in range(25):
        print(f'i: {i} - q = {RESULT[i]}')

    print('------------------------------')
    print(f'minimum: {min(RESULT)}')
    print(f'srednia: {np.mean(RESULT)}')
    war = wariancja(RESULT, np.mean(RESULT))
    standard_deviation = sqrt(war)
    print(f'odchylenie standardowe: {standard_deviation}')
    parameters = [i for i in range(25)]
    plot = Plot(parameters, RESULT)
    plot.show()


if __name__ == "__main__":
    main()
