import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_gradient(x):
    return x * (1 - x)


def linear(data, w, b):
    result = data[0] * w[0] + data[1] * w[1] + b
    return sigmoid(result)


class OneLayerMLP:
    def __init__(self, train_data, test_data, learning_rate, momentum_constant):
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.w = np.random.random((2, 4))
        self.b = np.random.random(2)
        self.w_M = np.random.random(2)
        self.b_M = np.random.random(1)

    def forward(self, data):
        results = [data]
        result = (linear(data, self.w[0], self.b[0]), linear(data, self.w[1], self.b[1]))
        results.append(result)
        result = linear(result, self.w_M, self.b_M)
        results.append(result)
        return results

    def backward(self, e, results, data):
        self.w_M = self.learning_rate * e * sigmoid_gradient(results[-1]) * data[
                                                                            0:1] + self.momentum_constant * self.w_M
        self.b_M = self.learning_rate * e * sigmoid_gradient(results[-1]) + self.momentum_constant * self.b_M

    def train(self):
        i = 1
        for data in self.train_data:
            print("第" + str(i) + "次训练")
            results = self.forward(data)
            e = data[2] - results[-1]
            error = 0.5 * e * e
            self.backward(e, results, data)
            print("error:" + str(error))
            '''print(self.w)
            print(self.b)'''
            i = i + 1

    def test(self):
        right_num = 0
        total_num = 0
        for data in self.test_data:
            total_num = total_num + 1
            results = self.forward(data)
            if abs(results[-1] - data[2]) < 0.5:
                right_num = right_num + 1
        print("正确率：" + str(100 * right_num / total_num) + "%")

    def decision_boundary(self):
        a = 1


if __name__ == '__main__':
    train_data = np.loadtxt('two_spiral_train_data.txt')
    test_data = np.loadtxt('two_spiral_test_data.txt')
    learning_rate = 1
    momentum_constant = 1
    MLP = OneLayerMLP(train_data, test_data, learning_rate, momentum_constant)
    MLP.train()
    MLP.test()
