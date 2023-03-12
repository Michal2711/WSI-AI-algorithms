import numpy as np


class Layer:
    def __init__(self, input_size, output_size, activation_function, deriv_activation) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.deriv_activation = deriv_activation
        self.preActivation = None
        self.postActivation = None
        self.input = None
        # self.weights = np.random.normal(0, 1, (self.output_size, self.input_size))  # 0 -> 1?
        self.weights = np.random.uniform(-1, 1, (self.output_size, self.input_size))

    def query(self, input_list):
        input_list = np.append(input_list, 1)
        output = np.array(input_list, ndmin=2).T  # transponujemy macierz żeby można było zrobić później mnożenie macierzy, czyli przemnożenie wejść przez wage i sume
        output = np.dot(self.weights, output)  # tutaj jest mnozenie macierzy
        self.input = input_list
        output = output.flatten()  # zdejmuje wymiar
        self.preActivation = output
        output = self.activation_function(output)
        self.postActivation = output

    def update_weights(self, error, alpha):
        self.weights -= alpha*error


class Neural:
    def __init__(self, number_of_hidden_layers, hidden_size, input_size, output_size, activation_function, deriv_activation, alpha) -> None:
        self.hidden_size = hidden_size
        self.hidden_number = number_of_hidden_layers
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.deriv_activation = deriv_activation
        self.alpha = alpha
        self.layers = list()
        self.layers.append(Layer(self.input_size+1, self.hidden_size, activation_function, deriv_activation))
        for i in range(self.hidden_number):
            self.layers.append(Layer(self.hidden_size+1, self.hidden_size, activation_function, deriv_activation))
        self.layers.append(Layer(self.hidden_size+1, self.output_size, lambda x: x, lambda x: 1))

    def query(self, input):
        for layer in self.layers:
            layer.query(input)
            input = layer.postActivation

    def gradient_descent(self, error):
        for index, layer in reversed(list(enumerate(self.layers))):
            x = np.array(layer.deriv_activation(layer.preActivation), ndmin=2).T
            preActivError = error*x
            error = np.dot(layer.weights.T, preActivError)
            error = np.delete(error, -1, 0)
            weights_error = np.dot(preActivError, np.array(layer.input, ndmin=2))
            layer.update_weights(weights_error, self.alpha)

    def train(self, inputs, target):
        self.query(inputs)
        output = np.array(self.layers[-1].postActivation, ndmin=2).T
        target = np.array(target, ndmin=2).T
        error = 2*(output - target)
        self.gradient_descent(error)
        return (np.average(np.power(output-target, 2)))

    def test(self, input):
        for layer in self.layers:
            layer.query(input)
            input = layer.postActivation
        return self.layers[-1].postActivation
