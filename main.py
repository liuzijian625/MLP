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
        self.b_M_max = None
        self.w_max = None
        self.b_max = None
        self.w_M_max = None
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.w = np.random.random((32, 2))
        self.b = np.random.random(32)
        self.w_M = np.random.random(32)
        self.b_M = np.random.random(1)

    def forward(self, data):
        results = [data]
        result = []
        for i in range(32):
            result.append(linear(data, self.w[i], self.b[i]))
        results.append(np.array(result))
        result = linear(result, self.w_M, self.b_M)
        results.append(result)
        return results

    def backward(self, e, results):
        b_gradient = e * sigmoid_gradient(results[-1])
        b_m = self.b_M
        self.w_M = self.learning_rate * e * sigmoid_gradient(results[-1]) * results[
            -2] + self.momentum_constant * self.w_M
        self.b_M = self.learning_rate * b_gradient + self.momentum_constant * self.b_M
        for i in range(32):
            self.w[i] = self.learning_rate * sigmoid_gradient(results[-2][i]) * results[
                                                                                    -3][
                                                                                0:1] * b_gradient * b_m + self.momentum_constant * \
                        self.w[i]
            self.b[i] = self.learning_rate * sigmoid_gradient(
                results[-2][i]) * b_gradient * b_m + self.momentum_constant * self.b[i]

    def train(self):
        self.acc = 0
        for j in range(200):
            print("第" + str(j + 1) + "轮训练")
            i = 1
            for data in self.train_data:
                '''print("第" + str(i) + "次训练")'''
                results = self.forward(data)
                e = data[2] - results[-1]
                error = 0.5 * e * e
                self.backward(e, results)
                '''print("error:" + str(error))'''
                i = i + 1
            acc_now = self.test(self.test_data)
            if acc_now > self.acc:
                self.j=j
                self.acc=acc_now
                self.w_max = self.w
                self.b_max = self.b
                self.w_M_max = self.w_M
                self.b_M_max = self.b_M
        print("最好结果在第"+str(self.j+1)+"轮产生")

    def test(self, datas):
        right_num = 0
        total_num = 0
        for data in datas:
            total_num = total_num + 1
            results = self.forward(data)
            if abs(results[-1] - data[2]) < 0.5:
                right_num = right_num + 1
        print("正确率：" + str(100 * right_num / total_num) + "%")
        return 100 * right_num / total_num

    def decision_boundary(self):
        a = 1

    def test_prep(self):
        self.b_M = self.b_M_max
        self.b = self.b_max
        self.w = self.w_max
        self.w_M = self.w_M_max


if __name__ == '__main__':
    train_data = np.loadtxt('two_spiral_train_data.txt')
    test_data = np.loadtxt('two_spiral_test_data.txt')
    learning_rate = 1
    momentum_constant = 1
    MLP = OneLayerMLP(train_data, test_data, learning_rate, momentum_constant)
    MLP.train()
    MLP.test_prep()
    MLP.test(test_data)
