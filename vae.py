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
import pdb


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
    
    def loss(self, x, L=10):
        encoder_dist = self.encoder(x)
        encoder_mean = encoder_dist.mean
        encoder_var = encoder_dist.variance
        loss = -(encoder_var-1).sum()
        loss -= (encoder_mean**2).sum()
        loss += torch.log(encoder_var).sum()
        loss /= 2
        
        z_sample = encoder_dist.rsample((L,))
        
        decoder_dist = vae_model.decoder(z_sample)
        log_likelihood = decoder_dist.log_prob(x)
        loss += log_likelihood.sum()
        return -loss
    

class VAE_symmetric(VAE):
    
    def __init__(self, data_dim, latent_dim=1, bias_decoder=False, std_grad=False):
        super().__init__()
        self.bias_decoder = bias_decoder
        
        self.encoder_fc = nn.Linear(data_dim, latent_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.kl_div = nn.KLDivLoss(reduction='sum')
        
        self.decoder_fc = nn.Linear(latent_dim, data_dim, bias=bias_decoder)
        self.std = nn.Parameter(torch.ones(data_dim), requires_grad=std_grad)
    
    def encoder(self, x):
        z = self.sigmoid(self.encoder_fc(x))
        return z
    
    def decoder(self, z):
        mean = self.decoder_fc(z)
        cov_mat = torch.diag(self.std**2)
        dist = MultivariateNormal(mean, cov_mat)
        return dist
    
    def loss(self, x, p_bernoulli=0.5, epsilon=1e-8):
        z = self.encoder(x)
        log_prior_true = torch.log(torch.tensor([1-p_bernoulli, p_bernoulli]))
        prior_predicted = torch.hstack((1-z, z))
        loss = -self.kl_div(input=log_prior_true, target=prior_predicted).sum()
        
        decoder_dist0 = self.decoder(-1*torch.ones_like(z))
        decoder_dist1 = self.decoder(torch.ones_like(z))
        log_likelihood0 = decoder_dist0.log_prob(x)
        log_likelihood1 = decoder_dist1.log_prob(x)
        log_likelihood = torch.vstack((log_likelihood0, log_likelihood1)).T
        loss += (prior_predicted * log_likelihood).sum()
        return -loss
    
    def sample(self, size=None, p_bernoulli=0.5):
        if size is None:
            z = torch.bernoulli(p_bernoulli)
            z = z.reshape((1, 1))
        else:
            p_vec = torch.zeros(size) + p_bernoulli
            z = torch.bernoulli(p_vec)
            z = z.reshape((size, 1))
        z[z==0] = -1
        generation_dists = self.decoder(z)
        generation_data = generation_dists.sample()
        return generation_data

    
class VAE_known(VAE):
    
    def __init__(self, data_dim, latent_dim=2, bias_decoder=False, std_grad=True):
        super().__init__()
        self.bias_decoder = bias_decoder
        
        self.encoder_fc = nn.Linear(data_dim, latent_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
        self.decoder_fc = nn.Linear(latent_dim, data_dim, bias=bias_decoder)
        self.std = nn.Parameter(torch.ones(data_dim), requires_grad=std_grad)
    
    def encoder(self, x):
        z = self.softmax(self.encoder_fc(x))
        return z
    
    def decoder(self, z):
        mean = self.decoder_fc(z)
        cov_mat = torch.diag(self.std**2)
        dist = MultivariateNormal(mean, cov_mat)
        return dist
    
    def loss(self, x, p_bernoulli=0.5, epsilon=1e-8):
        z = self.encoder(x)
        prior_true = torch.tensor([1-p_bernoulli, p_bernoulli])
        loss = -torch.xlogy(z, z+epsilon).sum()
        loss += torch.xlogy(z, prior_true).sum()
        
        decoder_dist0 = self.decoder(torch.tensor([1., 0.]))
        decoder_dist1 = self.decoder(torch.tensor([0., 1.]))
        log_likelihood0 = decoder_dist0.log_prob(x)
        log_likelihood1 = decoder_dist1.log_prob(x)
        log_likelihood = torch.vstack((log_likelihood0, log_likelihood1)).T
        loss += (z * log_likelihood).sum()
        return -loss

class VAE_known_tanh(VAE):
    
    def __init__(self, data_dim, latent_dim=1):
        super().__init__()
        
        self.encoder_fc = nn.Linear(data_dim, latent_dim)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        
        self.decoder_fc = nn.Linear(latent_dim, data_dim, bias=False)
        self.std = nn.Parameter(torch.ones(data_dim))
    
    def encoder(self, x):
        z = self.tanh(self.encoder_fc(x))
        return z
    
    def decoder(self, z):
        mean = self.decoder_fc(z)
        cov_mat = torch.diag(self.std**2)
        dist = MultivariateNormal(mean, cov_mat)
        return dist
    
    def loss(self, x, p_bernoulli=0.5):
        z = vae_model.encoder(x)
        loss = (1-z)/2 + z*p_bernoulli
        loss = loss.sum()

        decoder_dist = vae_model.decoder(z)
        log_likelihood = decoder_dist.log_prob(x)
        loss += log_likelihood.sum()
        return -loss