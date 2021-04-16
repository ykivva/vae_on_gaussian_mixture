import numpy as np

from vae import VAE_vanilla, VAE_symmetric, VAE_known, VAE_known_tanh
from datasets import DataGenerator, DataGeneratorSymmetric
from datasets import DataGeneratorStatic, DataGeneratorSymmetricStatic

MAX_ITER = 2000

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
        
        "experiment4": {
            "data_dim": 1000,
            "latent_dim": 1,
            "std_grad": False,
            "epochs": MAX_ITER // (((1000*np.linspace(0.001, 0.05, num=50)).astype(int)) // 1),
            "batch_size": 1,
            "alphas": np.linspace(0.001, 0.05, num=50),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 1,
        },
        
        "experiment5": {
            "data_dim": 100,
            "latent_dim": 1,
            "std_grad": False,
            "epochs": MAX_ITER // (((100*np.linspace(0.1, 5, num=50)).astype(int)+7) // 8),
            "batch_size": 8,
            "alphas": np.linspace(0.1, 5, num=50),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 10,
        },
        
        "experiment6": {
            "data_dim": 100,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-4,
            "epochs": 10000 // (((100*np.linspace(0.01, 10, num=1000)).astype(int)+0) // 1),
            "batch_size": 1,
            "alphas": np.linspace(0.01, 10, num=1000),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 10,
        },
        
        "experiment7": {
            "data_dim": 100,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-4,
            "epochs": 10000 // (((100*np.linspace(0.5, 30, num=1000)).astype(int)+7) // 8),
            "batch_size": 8,
            "alphas": np.linspace(0.5, 30, num=1000),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 10,
        },
        
        "experiment8": {
            "data_dim": 100,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-3,
            "epochs": 5000 // (((100*np.linspace(1, 100, num=200)).astype(int)+63) // 64),
            "batch_size": 64,
            "alphas": np.linspace(1, 100, num=200),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 10,
        },
        
        "experiment9": {
            "data_dim": 100,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-3,
            "epochs": 5000 // (((100*np.linspace(0.1, 4, num=200)).astype(int)+63) // 64),
            "batch_size": 64,
            "alphas": np.linspace(0.1, 4, num=200),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 10,
        },
        
        "experiment11": {
            "data_dim": 50,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-3,
            "epochs": 2000 // (((50*np.linspace(0.02, 4, num=200)).astype(int)+15) // 16) + 1,
            "batch_size": 16,
            "alphas": np.linspace(0.02, 4, num=200),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 7,
        },
        "experiment12": {
            "data_dim": 50,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-3,
            "epochs": 2000 // (((50*np.linspace(3.5, 7, num=100)).astype(int)+15) // 16) + 1,
            "batch_size": 16,
            "alphas": np.linspace(3.5, 7, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 7,
        },
        
        "experiment13": {
            "data_dim": 50,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-3,
            "epochs": 2000 // (((50*np.linspace(4, 12, num=100)).astype(int)+63) // 64) + 1,
            "batch_size": 64,
            "alphas": np.linspace(4, 12, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 7,
        },
        
        "experiment14": {
            "data_dim": 256,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 3e-3,
            "epochs": 10000 // (((256*np.linspace(1, 12, num=100)).astype(int)+31) // 32) + 1,
            "batch_size": 32,
            "alphas": np.linspace(1, 12, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 16,
        },
        
        "experiment15": {
            "data_dim": 1024,
            "latent_dim": 1,
            "std_grad": False,
            "lr": 1e-3,
            "epochs": 100 // (((256*np.linspace(1, 12, num=100)).astype(int)+31) // 32) + 1,
            "batch_size": 32,
            "alphas": np.linspace(1, 12, num=100),
            "std": 1,
            "p_bernoulli": 0.5,
            "d": 32,
        },
    },
}