import sys, os, time
import random, math
import fire

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

import config
from bayes_optimal import RAMP_VAE_gaus_ez
from utils import elapsed

import pdb


def train(config_name="test", dataset_name="SymmetricStatic", seed=2021):
    dataclass = config.datasets[dataset_name]
    bayes_config = config.bayes_configs[config_name]
    
    data_dim = bayes_config.get("data_dim", 100)
    
    alphas = bayes_config.get('alphas', [1])
    p = bayes_config.get('p', 0.5)    
    std = bayes_config.get('std', 1)    
    
    max_iter = bayes_config.get("max_iter", 1000)
    lambda_ = bayes_config.get("lambda", 0.9)
    eps = bayes_config.get("eps", 1e-4)
    
    for alpha in alphas:
        # Set a fixed seed
        random.seed(seed)
        np.random.seed(seed)
        
        data_size = int(data_dim * alpha)
        
        # Set up the logger
        group = f"N:{data_dim} p:{p}"
        logger = wandb.init(project="vae_on_gaussian_mixture", entity="y-kivva", group=group, reinit=True)
        logger.config.update({
            "method": "bayes optimal",
            "data dim": data_dim,
            "p": p,
            "alpha": alpha,
            "data size": data_size,
        })
        logger.name = f"N:{data_dim}, alpha:{alpha}"
        
        # Initialize observed data
        dataset = dataclass(
            data_dim=data_dim, data_size=data_size, 
            std=std, p=p, d=data_dim**(1/2)
        )
        y = dataset.data.clone().detach()
        
        # Initialize Low-RAMP starting configuration
        u_init = dataset.u
        v_init = dataset.v
#         u_init = torch.randn(data_size) * 1e-3
#         v_init = torch.randn(data_dim) * 1e-3
        solver = RAMP_VAE_gaus_ez(
            data_size=data_size, data_dim=data_dim, 
            p=p, u_init=u_init, v_init=v_init
        )
        
        # Run the Low-RAMP
        error = solver.fit(
            y, lambda_=lambda_, max_iter=max_iter,
            eps=eps, logger=logger,
            v_true = dataset.v
        )
        print(error)
        
        logger.finish()

        
if __name__=="__main__":
    fire.Fire(train)