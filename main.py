import pandas as pd
import numpy as np


class OneLayerMLP:
    def __init__(self, train_data, test_data, learning_rate, ):
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.w = np.random.random(2)
        self.b = np.random.random(1)

    def forward(self):
        for data in self.train_data:
            result = data[0] * self.w[0] + data[1] * self.w[1] + self.b


if __name__ == '__main__':
    train_data = np.loadtxt('two_spiral_train_data.txt')
    test_data = np.loadtxt('two_spiral_test_data.txt')
    learning_rate = 1
    MLP = OneLayerMLP(train_data, test_data, learning_rate)
    MLP.forward()
