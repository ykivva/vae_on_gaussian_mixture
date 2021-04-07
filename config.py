import numpy as np

from vae import VAE_vanilla, VAE_symmetric, VAE_known, VAE_known_tanh
from datasets import DataGenerator, DataGeneratorSymmetric
from datasets import DataGeneratorStatic, DataGeneratorSymmetricStatic

MAX_ITER = 1000

vae_config = {
    "models": {
        "VAE_symmetric": VAE_symmetric,
    },
    
    "datasets": {
        "SymmetricStatic": DataGeneratorSymmetricStatic,
    },
    
    "train_configs": {
        "test_config": {
            "data_dim": 100,
            "latent_dim": 1,
            "std_grad": False,
            "epochs": 100,
            "batch_size": 32,
            "alphas": [0.5, 1, 2, 5],
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 1,
        },
        
        "experiment1": {
            "data_dim": 500,
            "latent_dim": 1,
            "std_grad": False,
            "epochs": MAX_ITER // (((31+500*np.linspace(0.1, 20, num=100)).astype(int)) // 32),
            "batch_size": 32,
            "alphas": np.linspace(0.1, 20, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 1,
        },
        
        "experiment2": {
            "data_dim": 1000,
            "latent_dim": 1,
            "std_grad": False,
            "epochs": MAX_ITER // (((31+1000*np.linspace(0.1, 1, num=100)).astype(int)) // 32),
            "batch_size": 32,
            "alphas": np.linspace(0.1, 1, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 1,
        },
        
        "experiment3": {
            "data_dim": 1000,
            "latent_dim": 1,
            "std_grad": False,
            "epochs": MAX_ITER // (((7+1000*np.linspace(0.01, 1, num=100)).astype(int)) // 8),
            "batch_size": 8,
            "alphas": np.linspace(0.01, 1, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 1,
        },
    },
}