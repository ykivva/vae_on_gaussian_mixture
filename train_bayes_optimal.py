from logging import error
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

import pdb


def train(config_name="test", dataset_name="SymmetricStatic"):
    dataclass = config.datasets[dataset_name]
    bayes_config = config.bayes_configs[config_name]
    
    seeds = bayes_config.get("seeds", [13])

    data_dim = bayes_config.get("data_dim", 100)
    
    alphas = bayes_config.get('alphas', [1])
    p = bayes_config.get('p', 0.5)    
    std = bayes_config.get('std', 1)    
    
    max_iter = bayes_config.get("max_iter", 1000)
    lambda_ = bayes_config.get("lambda", 0.9)
    eps = bayes_config.get("eps", 1e-4)

    cos_mean = []
    l2_mean = []

    id_wandb = wandb.util.generate_id()    
    for alpha in alphas:
        cos_dists = np.zeros(len(seeds))
        l2_dists = np.zeros(len(seeds))
        for idx, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)
            
            data_size = int(data_dim * alpha)
            
            # Set up the logger
            group = f"N:{data_dim} p:{p} alpha:{alpha}"
            logger = wandb.init(project="vae_on_gaussian_mixture", entity="y-kivva", group=group, reinit=True)
            logger.config.update({
                "method": "bayes optimal",
                "config_name": config_name,
                "data dim": data_dim,
                "p": p,
                "alpha": alpha,
                "data size": data_size,
                "dataset": dataset_name,
            })
            logger.name = f"Bayes N:{data_dim};alpha:{alpha};seed:{seed}"
            
            # Initialize observed data
            dataset = dataclass(
                data_dim=data_dim, data_size=data_size, 
                std=std, p=p, d=data_dim**(1/2)
            )
            y = dataset.data.clone().detach()
            
            # Initialize Low-RAMP starting configuration
            u_init = torch.randn(data_size) * 1e-1
            v_init = torch.randn(data_dim) * 1e-1
            solver = RAMP_VAE_gaus_ez(
                data_size=data_size, data_dim=data_dim, 
                p=p, u_init=u_init, v_init=v_init
            )
            
            # Run the Low-RAMP
            errors = solver.fit(
                y, lambda_=lambda_, max_iter=max_iter,
                eps=eps, logger=logger,
                v_true = dataset.v
            )
            
            logger.finish()


            cos_dist = errors["cos_v"]
            l2_dist = errors["l2_v"]

            cos_dists[idx] = abs(cos_dist)
            l2_dists[idx] = l2_dist
        
        logger = wandb.init(id=id_wandb, group=group, resume="allow")
        logger.name = f"Bayes:{config_name}; Mean res"
        cos_mean.append([alpha, cos_dists.mean()]) 
        l2_mean.append([alpha, l2_dists.mean()])
        
        
        cos_mean_table =  wandb.Table(data=cos_mean, columns = ["Alpha", "Cosine distance"])
        logger.log({"Cosine distance" : wandb.plot.line(cos_mean_table, "Alpha", "Cosine distance",
                title="Cosine distance between centroids")})
        
        l2_mean_table =  wandb.Table(data=l2_mean, columns = ["Alpha", "l2 distance"])
        logger.log({"l2 distance" : wandb.plot.line(l2_mean_table, "Alpha", "l2 distance",
                title="l2 distance between centroids")})
        logger.finish()



        
if __name__=="__main__":
    fire.Fire(train)