import sys
from random import shuffle, random

import numpy as np


def load_file(file_name):
    data = []
    # Sex feature to numerical index dictionary
    sex_to_index = {'M': 0, 'F': 1, 'I': 2}
    # Read file
    with open(file_name, 'r') as file:
        for line in file:
            # Split entry
            line = line.strip().split(',')
            line[0] = sex_to_index[line[0]]
            # Add to data
            data.append(np.array(line, dtype=np.float64))
    return np.array(data)


def load_labels(data):
    labels = np.loadtxt(data, dtype=np.float64)
    return labels

#
# def train_dev_split(train_data, train_labels):
#     # We split the original train file to 80% train and 20% validation data
#     # We will delete this after hyper-parameters tuning and use only test file
#     #train_x, dev_x, train_y, dev_y = train_test_split(train_data, train_labels, test_size=0.2)
#     return train_x, dev_x, train_y, dev_y


class Perceptron:
    def __init__(self, train_x, train_y, lr=0.01):
        self.train_x = train_x
        self.train_y = train_y
        self.lr = lr
        # self.w = np.random.uniform(-1, 1, [np.unique(train_y).size, train_x.shape[1]])
        self.w = np.zeros((np.unique(train_y).size, train_x.shape[1]))
        self.train()

    def train(self, epochs=1000):
        # self.train_x, self.train_y = np.random.shuffle(self.train_x, self.train_y, random_state=1)
        for e in range(epochs):
            for x, y in zip(self.train_x, self.train_y):
                # predict
                y_hat = np.argmax(np.dot(self.w, x))
                if y != y_hat:
                    self.w[int(y), :] += self.lr * x
                    self.w[y_hat, :] -= self.lr * x

    def predict(self, test_x):
        predictions = np.zeros(test_x.shape[0])
        for i, e in enumerate(test_x):
            predictions[i] = np.argmax(np.dot(self.w, e))
        return predictions


class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization

    def predict(self, inputs):
        classification = np.sign(np.dot(np.array(inputs), self.w) + self.b)

        return classification


class PA:
    def __init__(self, x):
        self.x = x

    def predict(self):
        print("run pa")


def run_perceptron(train_data, train_labels):
    perceptron = Perceptron(train_data, train_labels)
    prediction = perceptron.predict(train_data)
    accuracy = np.mean(prediction == train_labels, dtype=np.float64)

    return accuracy


def main():
    args = sys.argv
    train_x, train_y, test_x = args[1], args[2], args[3]
    train_data = load_file(train_x)
    train_labels = load_labels(train_y)
    #train_x, dev_x, train_y, dev_y = train_dev_split(train_data, train_labels)
    perceptron_predic = run_perceptron(train_data, train_labels)
    print(perceptron_predic)


if __name__ == "__main__":
    main()


