import pandas as pd
import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_gradient(x):
    return x * (1 - x)


class OneLayerMLP:
    def __init__(self, train_data, test_data, learning_rate, momentum_constant):
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.w = np.random.random(2)
        self.b = np.random.random(1)

    def forward(self, data):
        result = data[0] * self.w[0] + data[1] * self.w[1] + self.b
        return sigmoid(result)

    def backward(self, e, result, data):
        self.w = self.learning_rate * e * sigmoid_gradient(result) * data[0:1] + self.momentum_constant * self.w
        self.b = self.learning_rate * e * sigmoid_gradient(result) + self.momentum_constant * self.b

    def train(self):
        i = 1
        for data in self.train_data:
            print("第" + str(i) + "次训练")
            result = self.forward(data)
            e = data[2] - result
            error = 0.5 * e * e
            self.backward(e, result, data)
            print("error:" + str(error))
            print(self.w)
            print(self.b)
            i = i + 1

    def test(self):
        right_num = 0
        total_num = 0
        for data in self.test_data:
            total_num = total_num + 1
            result = self.forward(data)
            if abs(result - data[2]) < 0.5:
                right_num = right_num + 1
        print("正确率：" + str(100 * right_num / total_num) + "%")

    def decision_boundary(self):
        a=1


if __name__ == '__main__':
    train_data = np.loadtxt('two_spiral_train_data.txt')
    test_data = np.loadtxt('two_spiral_test_data.txt')
    learning_rate = 1
    momentum_constant = 1
    MLP = OneLayerMLP(train_data, test_data, learning_rate, momentum_constant)
    MLP.train()
    MLP.test()
