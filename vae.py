import sys
import os
import math
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal


DEVICE = torch.device("cpu")


class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.compiled = False
        
    def compile(self, optimizer=None, **kwargs):
        if optimizer is not None:
            self.optimizer_class = optimizer
            self.optimizer_kwargs = kwargs
            self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        else:
            self.optimizer = None

        self.compiled = True
        self.to(DEVICE)
    
    def step(self, loss, train=True):
        self.zero_grad()
        self.optimizer.zero_grad()
        self.train(train)
        self.zero_grad()
        self.optimizer.zero_grad()
        
        loss.backward()
        self.optimizer.step()
        self.zero_grad()
        self.optimizer.zero_grad()
        
    def encoder(self, x):
        raise NotImplementedError()
        
    def decoder(self, z):
        raise NotImplementedError()
        

class VAE_vanilla(VAE):
    
    def __init__(self, data_dim, latent_dim=2, h_dim=100):
        super().__init__()
        self.encoder_fc1 = nn.Linear(data_dim, h_dim)
        self.encoder_fc2 = nn.Linear(h_dim, latent_dim)
        self.encoder_fc3 = nn.Linear(h_dim, latent_dim)
        
        self.decoder_fc1 = nn.Linear(latent_dim, h_dim)
        self.decoder_fc2 = nn.Linear(h_dim, data_dim)
        self.decoder_fc3 = nn.Linear(h_dim, data_dim)
        
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
    
    def encoder(self, x):
        h = self.tanh(self.encoder_fc1(x))
        mean = self.encoder_fc2(h)
        var = self.softplus(self.encoder_fc3(h))
        cov_mat = torch.diag_embed(var)
        dist = MultivariateNormal(mean, cov_mat)
        return dist
    
    def decoder(self, z):
        h = self.tanh(self.decoder_fc1(z))
        mean = self.decoder_fc2(h)
        var = self.softplus(self.decoder_fc3(h))
        cov_mat = torch.diag_embed(var)
        dist = MultivariateNormal(mean, cov_mat)
        return dist
    

def vae_vanilla_loss(x, vae_model, L=10):
    encoder_dist = vae_model.encoder(x)
    encoder_mean = encoder_dist.mean
    encoder_var = encoder_dist.variance
    loss = -(encoder_var - 1).sum()
    loss -= (encoder_mean**2).sum()
    loss += torch.log(encoder_var).sum()
    loss /= 2
    
    z_sample = encoder_dist.rsample((L,))
    
    decoder_dist = vae_model.decoder(z_sample)
    log_likelihood = decoder_dist.log_prob(x)
    loss += log_likelihood.sum()
    return -loss


class VAE_known(VAE):
    def __init__(self, data_dim, latent_dim=1, d=1, beta=100.):
        super().__init__()
        self.d = 1
        self.beta = beta
        
        self.encoder_fc = nn.Linear(data_dim, latent_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
        self.decoder_fc = nn.Linear(latent_dim, data_dim, bias=False)
        self.variance = nn.Parameter(torch.ones(data_dim))
    
    def encoder(self, x):
        z = self.tanh(self.beta*self.encoder_fc(x))
        return z
    
    def decoder(self, z):
        mean = self.decoder_fc(z)
        cov_mat = torch.diag(self.variance)
        dist = MultivariateNormal(mean, cov_mat)
        return dist


def vae_known_loss(x, vae_model, p_bernoulli=0.5):
    z = vae_model.encoder(x)
    loss = (1-z)/2 + z*p_bernoulli
    loss = loss.sum()
    
    decoder_dist = vae_model.decoder(z)
    log_likelihood = decoder_dist.log_prob(x)
    loss += log_likelihood.sum()
    return -loss