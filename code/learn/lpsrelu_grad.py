# lpsrelu_grad.py
"""
Created on Tue May 29 18:33:43 2018

@author:  Wentao Huang
"""
#import ipdb
import torch as tc
from .grad import Grad

class LPSReLUGrad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=1.0, 
                isorth=True, eps=1e-6, margin=-1.0, alpha=0.8):
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
            f.sigmoid_()
            g = f.mul(alpha).add_(1.0-alpha+eps)
            f.mul_(1.0-f).div_(g)
            dQ = dQ - alpha*beta/Num*X.t().mm(f)
            obj0 += -1.0/Num*g.log_().sum()
            if bias_requires_grad:
                db = db - alpha*beta/Num*f.sum(0)
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
        argnum = tc.tensor([8])
        ctx.save_for_backward(dC, db, argnum)
        return obj0 + obj1

