import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random, sys, os, json, math, time

import torch
from torchvision import datasets, transforms, utils
import visdom

import IPython
import pdb


def elapsed(last_time=[time.time()]):
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff


class BaseLogger(object):
    
    def __init__(self, name, verbose=True, max_capacity=int(1e5)):
        self.name = name
        self.data = {}
        self.running_data = {}
        self.reset_running = {}
        self.verbose = verbose
        self.max_capacity = max_capacity
        self.hooks = []
        
    def add_hook(self, hook, feature='epoch', freq=1):
        self.hooks.append((hook, feature, freq))
    
    def update(self, feature, x):
        if isinstance(x, torch.Tensor):
            x = x.clone().detach().cpu().numpy().mean()
        else:
            x = torch.tensor(x).data.cpu().numpy().mean()
            
        self.data[feature] = self.data.get(feature, [])
        feature_capacity = len(self.data[feature])
        if feature_capacity > self.max_capacity:
            self.data[feature] = self.data[feature][:feature_capacity//2]
            
        if feature not in self.running_data or self.reset_running.pop(feature, False):
            self.running_data[feature] = []
        self.running_data[feature].append(x)
        
        for hook, hook_feature, freq in self.hooks:
            if feature == hook_feature and len(self.running_data[feature]) % freq == 0:
                self.reset_running[feature] = True
                self.data[feature].append(np.mean(self.running_data[feature]))
                hook(self, self.data)
    
    def step(self):
        buf = ""
        buf += f"({self.name}) "
        for feature in self.running_data.keys():
            if len(self.running_data[feature]) == 0: continue
            val = np.mean(self.running_data[feature])
            if float(val).is_integer():
                buf += f"{feature}: {int(val)}, "
            else:
                buf += f"{feature}: {val:0.4f}, "
            self.reset_running[feature] = True
        buf += f" ... {elapsed():0.2f} sec"
        self.text(buf)
        
    def text(self, text, end="\n"):
        raise NotImplementedError()

    def plot(self, data, plot_name, opts={}):
        raise NotImplementedError()

    def images(self, data, image_name):
        raise NotImplementedError()

    def plot_feature(self, feature, opts={}):
        self.plot(self.data[feature], feature, opts)

    def plot_features(self, features, name, opts={}):
        stacked = np.stack([self.data[feature] for feature in features], axis=1)
        self.plot(stacked, name, opts={"legend": features})
        

class Logger(BaseLogger):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def text(self, text, end='\n'):
        print(text, end=end, flush=True)
        
    def plot(self, data, plot_name, opts={}):
        feature = opts.get("feature", None)
        if feature is not None:
            ydata = data[feature]
        else:
            ydata = data
        xlabel = opts.get("xlabel", None)
        ylabel = opts.get("ylabel", None)
        xdata = opts.get("xdata", None)
        title = opts.get("title", plot_name)
        
        fig = plt.figure()
        fig.clear()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        
        if xdata is None:
            plt.plot(ydata)
        else:
            plt.plot(xdata, ydata)
        plt.show()