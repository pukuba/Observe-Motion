from matplotlib import pyplot as plt
import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


weight = 1
sigmoid_x = np.arange(-10, 10)
sigmoid_y = sigmoid(weight * sigmoid_x)
plt.plot(sigmoid_x, sigmoid_y, label='test1')
plt.show()
