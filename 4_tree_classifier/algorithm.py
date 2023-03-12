import math
import numpy as np


class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None
        self.height = None


class TreeClassifier:
    def __init__(self, D, Y, arguments_names, max_height):
        self.D = D  # data ( x1, x2, x3, x4 )
        self.Y = Y  # classes ( target )
        self.arguments_names = arguments_names  # feature_names
        self.max_height = max_height
        self.YClasses = list(set(Y))  # uique class [ 0, 1, 2 ]
        self.YClassesCount = [list(Y).count(x) for x in self.YClasses]  # counter how many occurrences of the class there are in Y
        self.max_count_of_Y = np.bincount(Y).argmax()  # which class are most common in Y
        self.node = Node()
        self.node.height = 1  # height of the node
        self.entropy = self.get_entropy([x for x in range(len(self.Y))])
        self.best = list()

    def ID3(self):
        x_ids = [x for x in range(len(self.D))]  # unique data indexes
        arguments_ids = [x for x in range(len(self.arguments_names))]  # argument indexes
        self.node = self.id3_recv(x_ids, arguments_ids, self.node, 1)  # define first node

    def get_entropy(self, x_ids):
        Y = [self.Y[i] for i in x_ids]  # sorted Y by id
        Y_count = [Y.count(x) for x in self.YClasses]  # number of instance of each classes
        # calculate entropy for each category and sum
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in Y_count])

        return entropy

    def get_information_gain(self, x_ids, argument_id):
        info_gain = self.get_entropy(x_ids)  # total entropy I(U)
        d_arguments = [self.D[x][argument_id] for x in x_ids]  # chosen argument values
        argument_vals = list(set(d_arguments))  # unique chosen argument values
        argument_value_count = [d_arguments.count(x) for x in argument_vals]  # freguency of all values
        argument_value_id = [
            [x_ids[index]
            for index, x in enumerate(d_arguments)
            if x == y]
            for y in argument_vals
        ]
        # we get all indices where a particular x is located
        # example: [[0], [1, 2, 3], [4], [5], [6], [7, 8, 9], [10], [11, 12], [13, 14, 15, 16, 17], [18, 19]]
        # compute the information gain with the chosen argument
        info_gain_argument = sum([value_counts / len(x_ids) * self.get_entropy(value_ids)
                                    for value_counts, value_ids in zip(argument_value_count, argument_value_id)])

        info_gain = info_gain - info_gain_argument

        return info_gain

    def get_argument_max_information_gain(self, x_ids, argument_ids):
        # get the entropy for each atribute
        arguments_entropy = [self.get_information_gain(x_ids, argument_id) for argument_id in argument_ids]
        max_id = argument_ids[arguments_entropy.index(max(arguments_entropy))]  # get atribute id with max information gain

        return self.arguments_names[max_id], max_id

    def id3_recv(self, x_ids, argument_ids, node, height):
        if not node:
            node = Node()
        Y_in_arguments = [self.Y[x] for x in x_ids]  # sorted Y_in_arguments by id
        if len(set(Y_in_arguments)) == 1:
            node.value = Y_in_arguments[0]
            return node
        if len(argument_ids) == 0:
            node.value = max(set(Y_in_arguments), key=Y_in_arguments.count)
            return node
        # get atribute with maximizes the information gain
        best_argument_name, best_argument_id = self.get_argument_max_information_gain(x_ids, argument_ids)
        self.best.append(best_argument_id)
        node.value = best_argument_name
        node.childs = []
        argument_values = list(set([self.D[x][best_argument_id] for x in x_ids]))  # all argument values

        for value in argument_values:

            child = Node()
            child.value = value
            child.height = height + 1
            node.childs.append(child)
            if child.height == self.max_height:
                return node
            child_x_ids = [x for x in x_ids if self.D[x][best_argument_id] == value]
            if not child_x_ids:
                child.next = max(set(Y_in_arguments), key=Y_in_arguments.count)
            else:
                if best_argument_id in argument_ids:
                    to_remove = argument_ids.index(best_argument_id)
                    argument_ids.pop(to_remove)

                child.next = self.id3_recv(child_x_ids, argument_ids, child.next, child.height)
        return node
