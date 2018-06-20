# relu_grad.py
"""
Created on Tue May 29 21:19:33 2018

@author: Wentao Huang
"""
#import ipdb
import torch as tc
from .grad import Grad

class ReLUGrad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=1.0, 
                isorth=True, eps=1e-3, margin=-0.5):
        if isinstance(bias, tc.Tensor):
            bias_requires_grad = bias.requires_grad
        else:
            bias_requires_grad = False
        if bias is not None:
            if margin is not None:
                bias = bias - margin
        else:
            if margin is not None:
                bias = -margin
        Num = len(input)
        input.reiter()
        obj0 = 0.0
        obj1 = 0.0
        db = 0.0 if bias_requires_grad else None#tc.Tensor([0.0])
        dQ = 0.0
        for X, _ in input:
#            ipdb.set_trace()
            f = X.mm(C)
            if bias is not None:
                f.add_(bias)
            f.sign_().mul_(0.5).add_(0.5+eps)
            obj0 += -1.0/Num*f.log().sum()
            f.mul_(-1.0).add_(1.0+eps)
            dQ = dQ - 1.0/Num*X.t().mm(f)
            if bias_requires_grad:
                db = db - 1.0/Num*f.sum(0)
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
        argnum = tc.tensor([7])
        ctx.save_for_backward(dC, db, argnum)
#        ipdb.set_trace()
        return obj0 + obj1


