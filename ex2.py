import sys
from random import shuffle, random

import numpy as np
from sklearn.model_selection import train_test_split

perceptron_lr = 0.0001
percptron_epochs = 500
svm_lr = 0.0001
svm_lamda = 0.01
svm_epochs = 100
pa_lr = 0.0001
pa_epochs = 1000


def load_file(file_name):
    data = []
    # Sex feature to numerical index dictionary
    sex_to_index = {'M': -1.5, 'F': 1.5, 'I': 0}
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


def random_permutation(train_x, train_y):
    permute = np.random.permutation(train_x.shape[0])
    return train_x[permute], train_y[permute]


class Perceptron:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.lr = perceptron_lr
        # self.eta = eta
        # self.w = np.zeros((num_of_classes, len(train_x[0])))
        self.w = np.zeros((np.unique(train_y).size, train_x.shape[1]))
        self.train()

    def train(self, epochs=percptron_epochs):
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
    def __init__(self, train_x, train_y, num_of_classes=3, lamda=svm_lamda, lr=svm_lr):
        self.train_x = train_x
        self.train_y = train_y
        self.lr = lr
        # self.eta = eta
        self.lamda = lamda
        self.num_of_classes = num_of_classes
        self.w = np.random.uniform(-0.5, 0.5, [np.unique(train_y).size, train_x.shape[1]])
        # self.w = np.zeros((np.unique(train_y).size, train_x.shape[1]))
        self.train()

    def train(self, epochs=100):
        eta_lamda = 1 - self.lr * self.lamda
        for e in range(epochs):
            for x, y in zip(self.train_x, self.train_y):
                # predict
                y_hat = int(np.argmax(np.dot(self.w, x)))
                if y != y_hat:
                    self.w[int(y), :] = eta_lamda * self.w[int(y), :] + self.lr * x
                    self.w[y_hat, :] = eta_lamda * self.w[y_hat, :] - self.lr * x
                    for i in range(self.num_of_classes):
                        if i != y and i != y_hat:
                            self.w[i, :] *= eta_lamda
                    # eta_x = self.eta * x
                    # self.w[int(y), :] *= eta_lamda
                    # self.w[int(y), :] += eta_x
                    # self.w[y_hat, :] *= eta_lamda
                    # self.w[y_hat, :] -= eta_x
            # prediction_train = self.predict(self.train_x)
            # error = np.mean(prediction_train != self.train_y, dtype=np.float64)
            # print(1 - error)

    def predict(self, test_x):
        predictions = np.zeros(test_x.shape[0])
        for i, e in enumerate(test_x):
            predictions[i] = np.argmax(np.dot(self.w, e))
        return predictions


class PA:
    def __init__(self, train_x, train_y, lr=pa_lr):
        self.train_x = train_x
        self.train_y = train_y
        self.lr = lr
        self.w = np.zeros((np.unique(train_y).size, train_x.shape[1]))
        self.train()

    def train(self, epochs=pa_epochs):
        for e in range(epochs):
            for x, y in zip(self.train_x, self.train_y):
                # predict
                y_hat = int(np.argmax(np.dot(self.w, x)))
                if y != y_hat:
                    t = (max(0, 1 - (np.dot(self.w[int(y)], x)) + (np.dot(self.w[int(y_hat)], x)))) / (
                            np.linalg.norm(x) ** 2)
                    tx = t * x
                    self.w[int(y), :] += tx
                    self.w[y_hat, :] -= tx

    def predict(self, test_x):
        predictions = np.zeros(test_x.shape[0])
        for i, e in enumerate(test_x):
            predictions[i] = np.argmax(np.dot(self.w, e))
        return predictions


def run_perceptron(train_data, train_labels, test_data, test_labels):
    perceptron = Perceptron(train_data, train_labels)
    prediction_train = perceptron.predict(test_data)
    error = np.mean(prediction_train != test_labels, dtype=np.float64)

    # print("train accuracy: {} %".format(100 - np.mean(np.abs(prediction_train - train_labels)) * 100))
    # print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))
    return 1 - error, prediction_train


def run_svm(train_data, train_labels, test_data, test_labels):
    svm = SVM(train_data, train_labels)
    prediction_train = svm.predict(test_data)
    error = np.mean(prediction_train != test_labels, dtype=np.float64)
    return 1 - error, prediction_train


def run_pa(train_data, train_labels, test_data, test_labels):
    pa = PA(train_data, train_labels)
    prediction_train = pa.predict(test_data)
    error = np.mean(prediction_train != test_labels, dtype=np.float64)
    return 1 - error, prediction_train


def normalize(data):
    for i in range(len(data[0])):
        min_arg = float(min(data[:, i]))
        max_arg = float(max(data[:, i]))
        if min_arg == max_arg:
            return 1
        for j in range(len(data)):
            old = data[j, i]
            data[j, i] = (float(data[j, i]) - min_arg) / (max_arg - min_arg)
        return data


def organize_features(data, types):
    col = len(types) - 1
    # adding col of zeros
    for i in range(col):
        data = np.c_[np.zeros(len(data)), data]
    for i in range(len(data)):
        for j in range(len(types)):
            if data[i][col] == types[j]:
                # delete the original sign
                data[i][col] = float(0)
                # light the true bit
                data[i][j] = float(1)
    return data


def load_data(train_data, train_label, test_data):
    train_data = np.genfromtxt(train_data, delimiter=',', dtype="|U5")
    train_label = np.genfromtxt(train_label, delimiter=",")
    test_data = np.genfromtxt(test_data, delimiter=",", dtype="|U5")
    # test_label = np.genfromtxt(test_label, delimiter=',')

    train_data = organize_features(train_data, ['M', 'F', 'I'])
    test_data = organize_features(test_data, ['M', 'F', 'I'])

    train_data = normalize(train_data)
    # train_x, test_x, train_y, test_y = train_test_split(train_data, train_label, test_size=0.2)
    # np.savetxt('test_x.txt', test_x, fmt='%s', delimiter=',')
    # np.savetxt('test_y.txt', test_y, fmt='%s', delimiter='\n')
    # np.savetxt('train_x_n.txt', test_x, fmt='%s', delimiter=',')
    # np.savetxt('train_y_n.txt', test_x, fmt='%s', delimiter='\n')
    test_data = normalize(test_data)

    return train_data, train_label, test_data


def data_print(perceptron_pred, svm_pred, pa_pred):
    for i in range(svm_pred.shape[0]):
        print('perceptron: {}, svm: {}, pa: {}'.format(str(int(perceptron_pred[i])), str(int(svm_pred[i])),
                                                       str(int(pa_pred[i]))))


def main():
    args = sys.argv
    train_x, train_y, test_x, test_y = args[1], args[2], args[3], args[4]
    # train_data, train_labels, test_train = load_data(train_x, train_y, test_x)
    train_data = load_file(train_x)
    train_labels = load_labels(train_y)
    train_data, train_labels = random_permutation(train_data, train_labels)
    test_data = load_file(test_x)
    test_labels = load_labels(test_y)

    perceptron_precision, perceptron_predict = run_perceptron(train_data, train_labels, test_data, test_labels)
    svm_precision, svm_predict = run_svm(train_data, train_labels, test_data, test_labels)
    pa_precision, pa_predict = run_pa(train_data, train_labels, test_data, test_labels)

    print(perceptron_precision)
    print(svm_precision)
    print(pa_precision)

    # data_print(perceptron_predict, svm_predict, pa_predict)


if __name__ == "__main__":
    main()
