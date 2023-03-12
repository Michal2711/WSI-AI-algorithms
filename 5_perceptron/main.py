from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import perceptron
import matplotlib.pyplot as plt


def main():
    data = load_digits()
    Y = data.target
    X = []
    images = data.images
    for image in images:
        img = image.flatten().flatten()
        X.append(img)

    X = np.array(X)

    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, test_size=0.2)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, test_size=0.5)

    First_Perceptron = perceptron.Neural(2, 64, 64, 10, func, deriv, 0.03)
    second_Perceptron = perceptron.Neural(2, 64, 64, 10, func, deriv, 0.003)

    q = list()
    for i in range(10):
        for data, target in zip(X_train, Y_train):
            target_1_n = convert_1_n(target)
            out = First_Perceptron.train(data, target_1_n)
            q.append(out)

    correct = 0
    correct_rand = 0
    all = len(X_test)

    for data, target in zip(X_test, Y_test):
        First_Perceptron.query(data)
        second_Perceptron.query(data)
        output = First_Perceptron.layers[-1].postActivation
        output_rand = second_Perceptron.layers[-1].postActivation

        max_value = max(output)
        max_id = np.where(output == max_value)
        if(max_id == target):
            correct += 1

        max_rand = max(output_rand)
        max_rand_id = np.where(output_rand == max_rand)
        if(max_rand_id == target):
            correct_rand += 1

    print(f' result: {correct*100/all}')
    print(f' result_random: {correct_rand*100/all}')

    plt.plot(q)
    plt.show()


def convert_1_n(target):
    target_list = list()
    for i in range(10):
        if i == target:
            target_list.append(1)
        else:
            target_list.append(0)
    return target_list


def convert_to_decimal(X):
    result = 0
    dec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for x, d in zip(X, dec):
        result += x*d
    return result


def func(vector):
    output = list()
    for x in vector:
        output.append(np.exp(x)/(1 + np.exp(x)))
    return output


def deriv(vector):
    output = list()
    for x in vector:
        output.append(np.exp(x)/np.power(1+np.exp(x), 2))
    return output


if __name__ == "__main__":
    main()
