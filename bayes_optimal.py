import numpy as np
import matplotlib.pyplot as plt

import os, sys, math, random
import time, itertools
import parse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class RAMP_bipartite():
    
    def __init__(
        self, dw, dw2, 
        data_size, latent_dim, data_dim,
        p=0.5, u_init=None, v_init=None,
        fu_in=None, fv_in=None,
    ):
        """Initialize the configuration of the Low-RAMP
        
        Args:
            dw: gradient of the observed data with respect to w_ij --- Nxn tensor 
            dw2: second order derivatives of the observed data with respect to w_ij --- Nxn tensor
            data_size: number of observed samples --- N
            latent_dim: dimesion of the latent variables v_i, u_i --- 3
            data_dim: dimession of the observed data --- n (or sometimes M)
        """
        self.data_size = data_size
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.p = 0.5
        
        self.dw = dw
        self.dw2 = dw2
        self.S = dw.clone().detach()
        self.R = dw**2 + dw2
        
        self.u = torch.zeros(data_size, latent_dim)
        self.v = torch.zeros(data_dim, latent_dim)
        if u_init is not None: self.u = u_init
        if v_init is not None: self.v = v_init
        
        self.u_old = torch.zeros(data_size, latent_dim)
        self.v_old = torch.zeros(data_dim, latent_dim)
        
        self.bu = torch.zeros(data_size, latent_dim, 1)
        self.bv = torch.zeros(data_dim, latent_dim, 1)
        
        self.au = torch.zeros(data_size, latent_dim, latent_dim)
        self.av = torch.zeros(data_dim, latent_dim, latent_dim)
                
        self.sigma_u = torch.zeros(data_size, latent_dim, latent_dim)
        self.sigma_v = torch.zeros(data_dim, latent_dim, latent_dim)
        
        self.fu_in = fu_in or self._fu_in
        self.fv_in = fv_in or self._fv_in
    
    def _fu_in(self, A, B):
        raise NotImplementedError()
    
    def _fv_in(self, A, B):
        raise NotImplementedError()
        
    def step(self):
        raise NotImplementedError()
        
    
class RAMP_VAE_gaus_ez(RAMP_bipartite):
    
    def __init__(
        self, data_size, data_dim,
        p=0.5, u_init=None, v_init=None,
    ):
        """Initialize the configuration of Low-RAMP.
        
        Args:
            obs_data: observed data (i.e. in the overleaf - Y)
            data_size: number of observed samples --- N
            data_dim: dimession of the observed data --- n (or M)
        """
        self.data_size = data_size
        self.data_dim = data_dim
        self.p = 0.5
        
        self.u = torch.zeros(data_size)
        self.v = torch.zeros(data_dim)
        if u_init is not None: self.u = u_init
        if v_init is not None: self.v = v_init
        
        self.u_old = torch.zeros(data_size)
        self.v_old = torch.zeros(data_dim)
        
        self.bu = torch.zeros(data_size)
        self.bv = torch.zeros(data_dim)
        
        self.au = torch.zeros(data_size)
        self.av = torch.zeros(data_dim)
                
        self.sigma_u = torch.zeros(data_size)
        self.sigma_v = torch.zeros(data_dim)
        
    def _fu_in(self, a, b):
        numerator = 2 * (self.p - 1)
        denominator = (torch.exp(2 * b) - 1)*self.p + 1
        res = numerator / denominator + 1
        return res
    
    def _fv_in(self, a, b):
        # if torch.any(a+1 <= 0):
        #     raise ValueError("It is not possible to compute fv_in!")
        res = b / (a+1)
        return res
    
    def _compute_bu(self, s, r):
        bu =  s @ self.v / math.sqrt(self.data_dim) - ((s**2) @ self.sigma_v) * self.u / self.data_dim
        return bu
    
    def _compute_bv(self, s, r):
        bv =  self.u @ s / math.sqrt(self.data_dim) - (self.sigma_u @ (s**2)) * self.v / self.data_dim
        return bv
    
    def _compute_au(self, s, r):
        au = (s**2) @ (self.v**2) - r @ (self.v**2 + self.sigma_v)
        au /= self.data_dim
        return au
    
    def _compute_av(self, s, r, eps=1e-3):
        av = (self.u**2)@(s**2) - (self.u**2 + self.sigma_u)@r
        av /= self.data_dim
        #av[av<=-1+eps] = -1+eps
        return av
    
    def _compute_u(self):
        u = self._fu_in(self.au, self.bu)
        return u
    
    def _compute_v(self):
        v = self._fv_in(self.av, self.bv)
        return v
    
    def _compute_sigma_u(self):
        res = 4 * (1-self.p) * self.p
        res /= (self.p * (torch.exp(self.bu) + (torch.exp(-self.bu))) + torch.exp(-self.bu))**2
        return res
    
    def _compute_sigma_v(self):
        return 1/(self.av+1)
    
    def fit(
        self, y, *, lambda_=1, max_iter=100, eps=1e-5, 
        logger=None, u_true=None, v_true=None 
    ):
        errors = {}
        errors["error"] = 100
        s = y
        r = y**2 - 1
        i = 0
        
        cos = nn.CosineSimilarity(dim=0)
        
        while errors["error"] > eps and i < max_iter:
            i += 1
            logs = {}
            
            errors["error"] = self.step(s, r, lambda_=lambda_)
            logs["Low-RAMP error"] = errors["error"]
            if v_true is not None:
                cos_dist_v = abs(cos(self.v, v_true))
                l2_dist_v = min(
                    torch.linalg.norm(self.v-v_true),
                    torch.linalg.norm(self.v+v_true)
                    )
                
                logs["Cosine distance V"] = cos_dist_v
                logs["l2_v"] = l2_dist_v
                errors["cos_v"] = cos_dist_v
                errors["l2_v"] = l2_dist_v
            if u_true is not None:
                u_accuracy = torch.mean((u_true==self.u).float())
                errors["accuracy_u"] = u_accuracy
                logs["Accuracy U"] = u_accuracy
            
            if logger is not None:
                logger.log(logs)
        
        return errors
            
            
    def step(self, s, r, lambda_=1):
        error = 0
        
        bu_new = self._compute_bu(s, r)
        au_new = self._compute_au(s, r)
        self.bu = lambda_*bu_new + (1 - lambda_)*self.bu
        self.au = lambda_*au_new + (1 - lambda_)*self.au
        self.u_old = self.u.clone().detach()
        self.u = self._compute_u()
        self.sigma_u = self._compute_sigma_u()
        
        bv_new = self._compute_bv(s, r)
        av_new = self._compute_av(s, r)
        self.bv = lambda_*bv_new + (1 - lambda_)*self.bv
        self.av = lambda_*av_new + (1 - lambda_)*self.av
        self.v_old = self.v.clone().detach()
        self.v = self._compute_v()
        self.sigma_v = self._compute_sigma_v()
        
        error += torch.norm(self.u - self.u_old) / self.data_size
        error += torch.norm(self.v - self.v_old) / self.data_dim
        
        return error
    