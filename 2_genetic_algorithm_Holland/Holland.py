from typing import Tuple
import numpy as np


def Holland(q, P_0, u, p_m, p_c, t_max) -> list:

    t = 0
    o = creating_o(P_0, q)

    top_o_index, top_o, top_x = find_top(o, P_0)
    summary = sum(o)
    o = [number / summary for number in o]

    while t < t_max:
        index_list = [i for i in range(u)]  # index_list potrzebny bo random.choice musi mieć tablicie jednowymiarowa
        r = np.random.choice(index_list, u, replace=True, p=o)
        R = [P_0[i] for i in r]
        M = KxM(R, u, p_m, p_c)
        o = creating_o(M, q)
        new_top_o_index, new_top_o, new_top_x = find_top(o, M)
        summary = sum(o)
        o = [number / summary for number in o]
        if new_top_o >= top_o:
            top_o = new_top_o
            top_x = new_top_x
            top_o_index = new_top_o_index
        P_0 = M
        t += 1

    x = list()
    individual_10 = list()
    for i in range(len(top_x)):
        x.append(top_x[i])
        if((i+1) % 5 == 0 and i != 0):
            individual_10.append(convert_to_10(x))
            x.clear()
    result = q(individual_10)
    return top_x, result


def convert_to_10(x) -> int:
    # x to np [0,0,0,0,0] Gray
    binary_x = [0, 0, 0, 0, 0]
    element = 1
    try:
        first = x.index(element)
    except ValueError:
        return -16
    for i in range(5):
        if (i <= first):
            binary_x[i] = x[i]
        else:
            binary_x[i] = 0 if binary_x[i-1] + x[i] > 1 else binary_x[i-1] + x[i]
    decimal = 0
    p = 0
    for i in range(4, -1, -1):
        decimal += binary_x[i]*pow(2, p)
        p += 1
    return decimal - 16


def find_top(o, P_0) -> Tuple[int, float, list]:
    top_o_index = o.index(max(o))
    top_o = o[top_o_index]
    top_x = P_0[top_o_index]

    return top_o_index, top_o, top_x


def creating_o(P_0, q) -> list:
    o = list()
    x = list()
    individual_10 = list()
    for individual in P_0:
        for i in range(len(individual)):
            x.append(individual[i])
            if((i+1) % 5 == 0 and i != 0):
                individual_10.append(convert_to_10(x))
                x.clear()
        o.append(-q(individual_10))
        individual_10.clear()
    o = [number - min(o) + 1 for number in o]

    return o


def KxM(R, u, p_m, p_c):
    K = list()
    first = list()
    second = list()
    for i in range(0, u - 1, 2):
        a = R[i]
        b = R[i+1]
        if np.random.random_sample() < p_c:
            pos = np.random.randint(1, len(R[i]))
            a_end = a[pos::]
            first.extend(a[:pos:])
            first.extend(b[pos::])
            second.extend(b[:pos:])
            second.extend(a_end)
            K.append(first)
            K.append(second)
            first = list()
            second = list()
        else:
            K.append(a)
            K.append(b)

    M = list()
    n = 0
    for k in range(len(K)):
        individual_k = K[k]
        for i in range(len(individual_k)):
            if np.random.random_sample() < p_m:
                individual_k[i] = 1 if individual_k[i] == 0 else 0

        M.append(individual_k)
        n += 1

    return M
