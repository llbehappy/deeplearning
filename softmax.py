import numpy as np
import matplotlib.pyplot as plt
from __future__ import print_function


scores = [[3.0, 1.0, 2.0],
          [1.3, 4.0, 2.7]]


def softmax(x):
    """
    Purpose: compute softmax value for x
    :param x:
    :return:
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

# plot

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
print(scores)
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
