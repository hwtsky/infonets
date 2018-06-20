# tanh_grad.py
"""
Created on Tue May 29 13:14:16 2018

@author: Wentao Huang
"""
#import ipdb
import torch as tc
from .grad import Grad

class TanhGrad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=0.9, 
                isorth=True, eps=1e-6):
        if isinstance(bias, tc.Tensor):
            bias_requires_grad = bias.requires_grad
        else:
            bias_requires_grad = False
        if bias is not None:
            bias = beta * bias
        C1 = beta * C
        Num = len(input)
        input.reiter()
        obj0 = 0.0
        obj1 = 0.0
        db = 0.0 if bias_requires_grad else None#tc.Tensor([0.0])
        dQ = 0.0
        for X, _ in input:
#            ipdb.set_trace()
            f = X.mm(C1)
            if bias is not None:
                f.add_(bias)
            f.tanh_()
            obj0 += -1.0/Num*f.pow(2).mul_(-1.0).add_(eps + 1.0).log_().sum()
            dQ = dQ + 2.0*beta/Num*X.t().mm(f)
            if bias_requires_grad:
                db = db + beta/Num*f.sum(0)
        K = C.size(1)
        if K == 1:
            G = C.t().mm(C) + eps
            obj1 = -0.5*G.log()
        else:
            G = C.t().mm(C) + tc.diag(tc.full((C.size(1),), eps))
            sign, logdet = tc.slogdet(G)
            obj1 = -0.5*logdet
        dQ = dQ.mm(C.t().mm(C))
        if isorth:
            dC = dQ - C.mm(dQ.t()).mm(C)
        else:
            dC = dQ - C
        argnum = tc.tensor([6])
        ctx.save_for_backward(dC, db, argnum)
#        ipdb.set_trace()
        return obj0 + obj1

