# test_fun.py
"""
Created on Sat May  5 11:03:53 2018

@author: Wentao Huang
"""
import matplotlib.pyplot as plt
import torch as tc
import infonets as it


if __name__ == "__main__":

    x = tc.range(-3, 3, 0.1)
    beta = 5.0
    out = it.nn.fun.lpstanh_grad(x, beta, 1.0)#_grad+2, 0.9, margin=-0.5, negative_slope=2.5
    out1 = it.nn.fun.lpstanh_grad(x, beta, -1.0)
    out1 = it.nn.fun.fun_exe(None, 'lpstanh_grad', x, beta, -1.0)
    plt.figure(figsize=(6,6))
    plt.plot(x.numpy(), out.numpy(), 
             x.numpy(), out1.numpy(), 'b',
             x.numpy(), 0*out.numpy(), 'r')
    plt.show()
    