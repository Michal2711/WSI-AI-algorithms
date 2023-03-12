from sklearn.datasets import load_iris
from algorithm import TreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    data = load_iris()

    D = data.data
    Y = data.target
    ile = 0
    arguments_names = data.feature_names

    results = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
        ]

    D_train, D_rem, Y_train, Y_rem = train_test_split(D, Y, test_size=0.33)
    D_valid, D_test, Y_valid, Y_test = train_test_split(D_rem, Y_rem, test_size=0.5)
    depth = 2
    max_results = 0
    best_depth = 0
    while depth <= 4:
        Tree = TreeClassifier(D_train, Y_train, arguments_names, depth)
        Tree.ID3()
        results = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        ile = 0
        for i in range(len(D_valid)):
            arguments = D_valid[i]
            result_class = get_class(Tree, Tree.node, arguments, 0)
            if result_class is None:
                result_class = Tree.max_count_of_Y
            if Y_valid[i] == result_class:
                ile += 1
            results[Y_valid[i]][result_class] += 1
        print(f'depth: {depth}, result: {ile*100/25}')
        plt.title('Correlation between expected value and outcome')
        sns.heatmap(results, annot=True)
        plt.show()
        if ile*100/25 > max_results:
            best_depth = depth
        depth += 1
    Tree = TreeClassifier(D_train, Y_train, arguments_names, best_depth)
    Tree.ID3()
    results = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    ile = 0
    for i in range(len(D_test)):
        arguments = D_test[i]
        result_class = get_class(Tree, Tree.node, arguments, 0)
        if result_class is None:
            result_class = Tree.max_count_of_Y
        if Y_test[i] == result_class:
            ile += 1
        results[Y_test[i]][result_class] += 1
    print(ile*100/25)
    plt.title('Correlation between expected value and outcome')
    sns.heatmap(results, annot=True)
    plt.show()


def get_class(Tree, node, arguments, i):

    if node.childs is None:
        if node.next is not None and node.next.value in [0, 1, 2]:
            return node.next.value
        elif node.next is None:
            return None
        else:
            return get_class(Tree, node.next, arguments, i)
    for x in node.childs:
        index = Tree.best[i]
        if x.value == arguments[index]:
            return get_class(Tree, x, arguments, i+1)


if __name__ == "__main__":
    main()
