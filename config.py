import numpy as np

from vae import VAE_vanilla, VAE_symmetric, VAE_known, VAE_known_tanh
from datasets import DataGenerator, DataGeneratorSymmetric
from datasets import DataGeneratorStatic, DataGeneratorSymmetricStatic

MAX_ITER = 2000


def get_num_epochs(max_iter, data_dim, alphas, batch_size):
    data_sizes = (data_dim * alphas).astype(int)
    data_sizes = (data_sizes + batch_size -  1) // batch_size
    num_epochs = max_iter // data_sizes
    num_epochs[num_epochs<1] = 1
    return num_epochs


def generate_vae_config(
    data_dim=1024, latent_dim=1,
    max_iter = 2000,
    std_grad=False, lr=1e-3,
    epochs=None, batch_size=64,
    alphas=np.array([3, 4, 5]), std=1,
    p=0.5, d=None, seeds=[13]
):
    if d is None: 
        d=data_dim**(1/2)
    if epochs is None:
        epochs = get_num_epochs(max_iter, data_dim, alphas, batch_size)
    
    config = {
        "data_dim": data_dim,
        "latent_dim": latent_dim,
        "std_grad": std_grad,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "alphas": alphas,
        "std": std,
        "p": p,
        "d": d,
        "seeds": seeds,
    }
    return config


def generate_bayes_config(
    data_dim=100, p=0.5,
    alphas=[1], lambda_=1,
    max_iter=1000, eps=1e-4, 
    std=1, seeds=[13]
):
    config = {
        "data_dim": data_dim, 
        "p": p,
        "alphas": alphas,
        "lambda": lambda_,
        "max_iter": max_iter,
        "eps": eps,
        "std": std,
        "seeds": seeds,
    }
    return config


# Dictionary with possible models for VAE
models = {
        "VAE_symmetric": VAE_symmetric,
    }

# Dictionary with datasets
datasets = {
        "SymmetricStatic": DataGeneratorSymmetricStatic,
    }

bayes_configs = {
    "bayes.v1": generate_bayes_config(
        data_dim=1024,
        p=0.5,
        alphas=np.linspace(1, 10, num=100),
        lambda_=0.95,
        max_iter=2000,
        eps=1e-6,
        std=1,
        seeds=[13, 26, 39, 666, 777, 1313, 2021, 6666, 7777, 9999]
    ),
}
        

vae_configs = {
    "vae.v1": generate_vae_config(
        data_dim=1024,
        latent_dim=1,
        std_grad=False,
        max_iter=4500,
        lr=3e-3,
        epochs=None,
        batch_size=512,
        alphas=np.linspace(1, 10, num=100),
        std=1,
        p=0.5,
        seeds=[13, 26, 39, 666, 777, 1313, 2021, 6666, 7777, 9999]
    ),
}