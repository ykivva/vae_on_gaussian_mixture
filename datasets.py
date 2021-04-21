import numpy as np
import matplotlib as mpl

import os, sys, math, random, tarfile, glob, time, yaml, itertools
import random
import parse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.distributions.multivariate_normal import MultivariateNormal
from torchvision import transforms, utils


class DataGenerator(IterableDataset):
    NUM_CENTROIDS = 2
    
    def __init__(self, data_dim, data_size, std=0.01, d=1, v=None, p_bernoulli=0.5):
        super().__init__()
        self.data_dim = data_dim
        self.data_size = data_size
        self.d = d
        self.std = std
        self.p_bernoulli = torch.tensor(p_bernoulli)
        self.v = v or torch.normal(0., 1., (2, data_dim))
    
    
    def __iter__(self, gen=None):
        worker_info = torch.utils.data.get_worker_info()
        gen = gen or self._generator
        if worker_info is None:
            iter_start = 0
            iter_end = self.data_size
        else:
            per_worker = int(math.ceil(self.data_size / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_data_size = min(self.data_size - worker_id*per_worker, per_worker)
        return gen(worker_data_size)
        
    def _generator(self, data_size):
        for _ in range(data_size): 
            y = torch.zeros(2)
            if torch.bernoulli(self.p_bernoulli)==1:
                y[0] = 1.
            else:
                y[1] = 1.
            yield y @ self.v/(self.d) + torch.normal(0, self.std, (self.data_dim,))
            
            
class DataGeneratorSymmetric(IterableDataset):
    
    def __init__(self, data_dim, data_size, std=0.01, d=1, v=None, p_bernoulli=0.5):
        super().__init__()
        self.data_dim = data_dim
        self.data_size = data_size
        self.d = d
        self.std = std
        self.p_bernoulli = torch.tensor(p_bernoulli)
        self.v = v or torch.normal(0., 1., (data_dim,))
    
    
    def __iter__(self, gen=None):
        worker_info = torch.utils.data.get_worker_info()
        gen = gen or self._generator
        if worker_info is None:
            iter_start = 0
            iter_end = self.data_size
        else:
            per_worker = int(math.ceil(self.data_size / float(worker_info.num_workers)))
            worker_id = worker_info.id
            worker_data_size = min(self.data_size - worker_id*per_worker, per_worker)
        return gen(worker_data_size)
        
    def _generator(self, data_size):
        for _ in range(data_size):
            if torch.bernoulli(self.p_bernoulli)==1:
                y = 1.
            else:
                y = -1.
            yield y * self.v/(self.d) + torch.normal(0, self.std, (self.data_dim,))
            

class DataGeneratorStatic(IterableDataset):
    NUM_CENTROIDS = 2
    
    def __init__(
        self, data_dim, data_size,
        std=1, d=1, v=None, p_bernoulli=0.5,
        gen=None, shuffle=True
    ):
        super().__init__()
        self.data_dim = data_dim
        self.data_size = data_size
        self.std = std
        self.d = d
        self.p_bernoulli = torch.tensor(p_bernoulli)
        self.v = v or torch.normal(0., 1., (self.NUM_CENTROIDS, data_dim))
        self.shuffle = shuffle
        self.data = []
        gen = gen or self._generator
        for data_sample in gen(data_size):
            self.data.append(data_sample)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.shuffle:
            random.shuffle(self.data)
        if worker_info is None:
            iter_start = 0
            iter_end = self.data_size
        else:
            per_worker = int(math.ceil(self.data_size / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start+per_worker, self.data_size)
        return iter(self.data[iter_start:iter_end])
        
        
    def _generator(self, data_size):
        for _ in range(data_size):
            y = torch.zeros(2)
            if torch.bernoulli(self.p_bernoulli)==1:
                y[1] = 1.
            else:
                y[0] = 1.
            yield y @ self.v/(self.d) + torch.normal(0, self.std, (self.data_dim,))
    

class DataGeneratorSymmetricStatic(IterableDataset):
    
    def __init__(
        self, data_dim, data_size, 
        std=1, d=1, v=None, p_bernoulli=0.5,
        gen=None, shuffle=True
    ):
        super().__init__()
        self.data_dim = data_dim
        self.data_size = data_size
        self.std = std
        self.d = d
        self.p_bernoulli = torch.tensor(p_bernoulli)
        self.v = v or torch.normal(0., 1., (data_dim, ))
        self.shuffle = shuffle
        self.data = []
        gen = gen or self._generator
        for data_sample in gen(data_size):
            self.data.append(data_sample)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if self.shuffle:
            random.shuffle(self.data)
        if worker_info is None:
            iter_start = 0
            iter_end = self.data_size
        else:
            per_worker = int(math.ceil(self.data_size / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start+per_worker, self.data_size)
        return iter(self.data[iter_start:iter_end])
        
        
    def _generator(self, data_size):
        for _ in range(data_size):
            if torch.bernoulli(self.p_bernoulli)==1:
                y = 1.
            else:
                y = -1.
            yield y * self.v/(self.d) + torch.normal(0, self.std, (self.data_dim,))
            
            
def generate_multiv_gauss(n_samples, mean, cov_mat):
    dist = MultivariateNormal(mean, cov_mat)
    return dist.sample((n_samples,))