import sys, os, time, math
import functools

import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets


def visualize_distributions(data2vis, labels, colors=[], alpha=0.1):
    plt.figure()
    colors = ['navy', 'darkorange']

    for color, i in zip(colors, [0, 1]):
        plt.scatter(data_proj[labels==i, 0], data_proj[labels==i, 1], 
                    color=color,
                    alpha=0.1)

    plt.show()