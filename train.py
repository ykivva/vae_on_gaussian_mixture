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
from utils import elapsed

import pdb


def train(
    config_name="test_config",
    vae_dataset_name="SymmetricStatic",
    vae_model_name="VAE_symmetric",
    lr_decay=False
):
    
    model_class = config.models[vae_model_name]
    dataclass = config.datasets[vae_dataset_name]
    vae_config = config.vae_configs[config_name]
    
    data_dim = vae_config.get('data_dim', 100)
    latent_dim = vae_config.get('latent_dim', 1)
    std_grad = vae_config.get("std_grad", True)
    lr = vae_config.get("lr", 3e-3)
    
    epochs_list = vae_config.get('epochs', 50)
    batch_size_ = vae_config.get('batch_size', 32)
    alphas = vae_config.get('alphas', [1])
    seeds = vae_config.get('seeds', [13])
    
    std = vae_config.get('std', 1)
    p = vae_config.get('p', 0.5)
    d = vae_config.get('d', 1)
    
    
    cos = nn.CosineSimilarity(dim=0)
    cos_mean = []
    l2_mean = []

    id_wandb = wandb.util.generate_id()
    for alpha, epochs in zip(alphas, epochs_list):
        cos_dists = np.zeros(len(seeds))
        l2_dists = np.zeros(len(seeds))
        for idx, seed in enumerate(seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.manual_seed(seed)

            data_size = int(data_dim * alpha)

            group = f"N:{data_dim} p:{p} alpha:{alpha}"
            logger = wandb.init(project="vae_on_gaussian_mixture", group=group, reinit=True)
            wandb_config = {
                "method": "VAE",
                "epochs": epochs,
                "data_dim": data_dim,
                "alpha": alpha,
                "seed": seed,
                "p": p,
                "config_name": config_name,
                "dataset": vae_dataset_name,
                "datasize": data_size,
            }
            logger.config.update(wandb_config)
            logger.name = f"VAE N:{data_dim};alpha:{alpha};seed:{seed}"

            batch_size = min(batch_size_, data_size)
            
            model = model_class(data_dim=data_dim, latent_dim=latent_dim, std_grad=std_grad)
            
            dataset = dataclass(data_dim=data_dim, data_size=data_size,
                            std=std, d=d, p=p)

            cur_lr = lr
            model.compile(torch.optim.Adam, lr=cur_lr, amsgrad=True)
            data_loader = DataLoader(dataset, batch_size=batch_size)
            
            cur_iter = 0
            for _ in range(epochs):
                for data_batch in data_loader:
                    loss = model.loss(data_batch)
                    model.step(loss)
                    logger.log({"loss":loss})
                    cur_iter += 1
                    if lr_decay and cur_iter>1500:
                        cur_iter=0
                        cur_lr /= 3
                        model.compile(torch.optim.Adam, lr=cur_lr, amsgrad=True)
            
            # num_vis_dots=50
            # real_data = np.array([x.numpy() for x in dataset.data])
            # num_choices = min(len(real_data), num_vis_dots)
            # choice = np.random.choice(len(real_data), size=num_choices, replace=False)
            # real_data = real_data[choice]
            
            # generated_data = model.sample(num_vis_dots)
            # data_vis = np.vstack((real_data, generated_data))
            # labels = np.ones(len(data_vis))
            # labels[:len(real_data)] = 0
            
            # pca = decomposition.PCA(n_components=2)
            # data_pca = pca.fit(data_vis)
            # data_proj = pca.transform(data_vis)
            
            # colors = ['navy', 'darkorange']
            # for color, i in zip(colors, [0, 1]):
            #     plt.scatter(data_proj[labels==i, 0], data_proj[labels==i, 1], 
            #                 color=color,
            #                 alpha=0.8)
            #     plt.title("Visualize distribution")
            # wandb.log({"Visualized distribution": plt})
            logger.finish()
            
            predicted_centroid = model.decoder_fc.state_dict()['weight'].squeeze()
            target_centroid = dataset.v / dataset.d
            cos_dist = cos(predicted_centroid, target_centroid)
            l2_dist = min(torch.linalg.norm(target_centroid-predicted_centroid),
                        torch.linalg.norm(target_centroid+predicted_centroid))
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