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
    

def elapsed(last_time=[time.time()]):
    """ Returns the time passed since elapsed() was last called. """
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff