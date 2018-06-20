# misc.py
"""
Created on Tue Jun  5 00:21:24 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division, print_function
#import ipdb
import numpy as np
import torch as tc


def get_b_beta_by_n_sgd(n=1, xmax=1.0e+4, Nx=1e+6, to_tensor=True, dtype=tc.float):
    b = np.log(1.0*n)
    x = np.linspace(-xmax, xmax, int(Nx))
    y0 = 1.0/(1.0 + np.exp(-x - b))
    y0 = n*(y0**n)*(1.0 - y0)
    v = np.trapz(y0, x)
    y0 = (x**2)*y0
    s = np.trapz(y0, x)
    beta = np.sqrt(s/v)
    if to_tensor:
        b = tc.tensor(b,dtype=dtype)
        beta = tc.tensor(beta,dtype=dtype)
    return b, beta

def get_b_beta_by_tanh(xmax=1.0e+4, Nx=1e+6, to_tensor=True, dtype=tc.float):
    b = 0.0
    x = np.linspace(-xmax, xmax, int(Nx))
    y0 = 1.0 - np.tanh(x)**2
    v = np.trapz(y0, x)
    y0 = (x**2)*y0
    s = np.trapz(y0, x)
    beta = np.sqrt(s/v)
    if to_tensor:
        b = tc.tensor(b,dtype=dtype)
        beta = tc.tensor(beta,dtype=dtype)
    return b, beta

def get_b_gamma_by_ReLU_margin(margin=1.0, xmax=1.0e+4, Nx=1e+6, 
                               to_tensor=True, dtype=tc.float):
    b = 0.0
    xm = 2.0 + np.exp(-margin)
    x = np.linspace(-xmax, xm, Nx)
    y0 = 1.0/(1.0 + np.exp(-x+margin))
    v = np.trapz(y0, x)
    y0 = (x**2)*y0
    s = np.trapz(y0, x)
    gamma = np.sqrt(s/v)# =1.52
    if to_tensor:
        b = tc.tensor(b,dtype=dtype)
        gamma = tc.tensor(gamma,dtype=dtype)
    return b, gamma

def get_b_gamma_by_ReLU_margin_1(margin=1.0, gamma=2.22, xmax=1.0e+4, 
                                 Nx=1e+6, to_tensor=True, dtype=tc.float):
    b = 0.0
    x = np.linspace(-xmax, xmax, Nx)
    y0 = 1.0/((1.0 + np.exp(-x+margin*gamma))*(1.0 + np.exp(x+margin*gamma)))
    v = np.trapz(y0, x)
    y0 = (x**2)*y0
    s = np.trapz(y0, x)
    gamma = np.sqrt(s/(v ))
    if to_tensor:
        b = tc.tensor(b,dtype=dtype)
        gamma = tc.tensor(gamma,dtype=dtype)
    return b, gamma

def get_b_gamma_by_ReLU_margin_2(margin=1.0, gamma=2.22, maxiters=10, xmax=1.0e+4,
                                 Nx=1e+6, to_tensor=True, dtype=tc.float):
    b, gamma = get_b_gamma_by_ReLU_margin_1(margin=margin, gamma=gamma, xmax=xmax, 
                                            Nx=Nx, to_tensor=to_tensor, dtype=to_tensor)
    for i in np.arange(maxiters):
        b, gamma = get_b_gamma_by_ReLU_margin_1(margin=margin, gamma=gamma, xmax=xmax, 
                                                Nx=Nx,to_tensor=to_tensor, dtype=to_tensor)
    return b, gamma


