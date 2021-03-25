import numpy as np
import matplotlib as mpl

import os, sys, math, random, tarfile, glob, time, yaml, itertools
import parse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import transforms, utils


class DataGenerator(IterableDataset):
    NUM_CENTROIDS = 2
    
    def __init__(self, data_dim, max_iter, epsilon=0.01, d=1, v=None, p_bernoulli=0.5):
        super().__init__()
        self.data_dim = data_dim
        self.max_iter = max_iter
        self.d = d
        self.epsilon = epsilon
        self.p_bernoulli = torch.tensor(p_bernoulli)
        self.v = v or torch.normal(0., 1., (2, data_dim))
    
    
    def __iter__(self, gen=None):
        worker_info = torch.utils.data.get_worker_info()
        gen = gen or self._generator
        if worker_info is not None:
            per_worker = int(math.ceil(self.max_iter / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_max_iter = min(self.max_iter - worker_id*per_worker, per_worker)
        return gen(worker_max_iter)
        
    def _generator(self, max_iter):
        for _ in range(max_iter):
            y = torch.zeros(2)
            if torch.bernoulli(self.p_bernoulli)==1:
                y[0] = 1.
            else:
                y[1] = 1.
            yield y @ self.v/(self.d) + torch.normal(0, self.epsilon, (self.data_dim,))
            
            
class DataGeneratorSymmetric(IterableDataset):
    NUM_CENTROIDS = 2
    
    def __init__(self, data_dim, max_iter, epsilon=0.01, d=1, v=None, p_bernoulli=0.5):
        super().__init__()
        self.data_dim = data_dim
        self.max_iter = max_iter
        self.d = d
        self.epsilon = epsilon
        self.p_bernoulli = torch.tensor(p_bernoulli)
        self.v = v or torch.normal(0., 1., (data_dim,))
    
    
    def __iter__(self, gen=None):
        worker_info = torch.utils.data.get_worker_info()
        gen = gen or self._generator
        if worker_info is not None:
            per_worker = int(math.ceil(self.max_iter / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_max_iter = min(self.max_iter - worker_id*per_worker, per_worker)
        return gen(worker_max_iter)
        
    def _generator(self, max_iter):
        for _ in range(max_iter):
            if torch.bernoulli(self.p_bernoulli)==1:
                y = 1.
            else:
                y = -1.
            yield y * self.v/(self.d) + torch.normal(0, self.epsilon, (self.data_dim,))
            
            
def generate_multiv_gauss(n_samples, mean, cov_mat):
    dist = MultivariateNormal(mean, cov_mat)
    return dist.sample((n_samples,))