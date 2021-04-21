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

from config import vae_config
from utils import elapsed

import pdb


def train(
    vae_model_name="VAE_symmetric",
    vae_dataset_name="SymmetricStatic",
    train_config="test_config",
    lr_decay=False
):
    
    model_class = vae_config["models"][vae_model_name]
    dataclass = vae_config["datasets"][vae_dataset_name]
    config = vae_config["train_configs"][train_config]
    
    data_dim = config.get('data_dim', 100)
    latent_dim = config.get('latent_dim', 1)
    std_grad = config.get("std_grad", True)
    lr = config.get("lr", 3e-3)
    
    epochs_list = config.get('epochs', 50)
    batch_size_ = config.get('batch_size', 32)
    alphas = config.get('alphas', [1])
    
    std = config.get('std', 1)
    p_bernoulli = config.get('p_bernoulli', 0.5)
    d = config.get('d', 1)
    num_vis_dots = 100
    
    
    cos = nn.CosineSimilarity(dim=0)
    cos_similarities = []
    l2_distnaces =[]
    id_wandb = wandb.util.generate_id()
    cos_dist_table = []
    l2_dist_table = []
    for alpha, epochs in zip(alphas, epochs_list):
        wandb.init(project="vae_on_gaussian_mixture", group=train_config, reinit=True)
        data_size = int(data_dim * alpha)
        batch_size = min(batch_size_, data_size)
        wandb.run.name = f"N:{data_dim};M:{data_size};alpha:{alpha}"
        
        model = model_class(data_dim=data_dim, latent_dim=latent_dim, std_grad=std_grad)
        
        dataset = dataclass(data_dim=data_dim, data_size=data_size,
                          std=std, d=d, p_bernoulli=p_bernoulli)

        cur_lr = lr
        model.compile(torch.optim.Adam, lr=cur_lr, amsgrad=True)
        data_loader = DataLoader(dataset, batch_size=batch_size)
        
        cur_iter = 0
        for epoch in range(epochs):
            for data_batch in data_loader:
                loss = model.loss(data_batch)
                model.step(loss)
                wandb.log({"loss":loss})
                cur_iter += 1
                if lr_decay and cur_iter>1500:
                    cur_iter=0
                    cur_lr /= 3
                    model.compile(torch.optim.Adam, lr=cur_lr, amsgrad=True)
        
        
        real_data = np.array([x.numpy() for x in dataset.data])
        num_choices = min(len(real_data), num_vis_dots)
        choice = np.random.choice(len(real_data), size=num_choices, replace=False)
        real_data = real_data[choice]
        
        generated_data = model.sample(num_vis_dots)
        data_vis = np.vstack((real_data, generated_data))
        labels = np.ones(len(data_vis))
        labels[:len(real_data)] = 0
        
        pca = decomposition.PCA(n_components=2)
        data_pca = pca.fit(data_vis)
        data_proj = pca.transform(data_vis)
        
        colors = ['navy', 'darkorange']
        for color, i in zip(colors, [0, 1]):
            plt.scatter(data_proj[labels==i, 0], data_proj[labels==i, 1], 
                        color=color,
                        alpha=0.8)
            plt.title("Visualize distribution")
        wandb.log({"Visualized distribution": plt})
        wandb.finish()
        
        wandb.init(id=id_wandb, group=train_config, resume="allow")
        
        predicted_centroid = model.decoder_fc.state_dict()['weight'].squeeze()
        target_centroid = dataset.v / dataset.d
        cos_dist = cos(predicted_centroid, target_centroid)
        l2_dist = min(torch.linalg.norm(target_centroid-predicted_centroid),
                      torch.linalg.norm(target_centroid+predicted_centroid))
        
        cos_dist_table.append([alpha, abs(cos_dist)]) 
        l2_dist_table.append([alpha, l2_dist])
        
        cos_table =  wandb.Table(data=cos_dist_table, columns = ["Alpha", "Cosine distance"])
        wandb.log({"Cosine distance" : wandb.plot.line(cos_table, "Alpha", "Cosine distance",
                   title="Cosine distance between centroids")})
        
        l2_table =  wandb.Table(data=l2_dist_table, columns = ["Alpha", "l2 distance"])
        wandb.log({"l2 distance" : wandb.plot.line(l2_table, "Alpha", "l2 distance",
                   title="l2 distance between centroids")})
        
        wandb.finish()

if __name__=="__main__":
    fire.Fire(train)