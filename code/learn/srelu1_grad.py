# srelu1_grad.py
"""
Created on Fri Jun  1 01:34:56 2018

@author: Wentao Huang
"""
#import ipdb
import torch as tc
from .grad import Grad

class SReLU1Grad(Grad):

    @staticmethod
    def forward(ctx, input, C, bias=None, beta=1.0, isorth=True, 
                eps=1e-6, ith_cls=0, R=None, margin=None):
        if isinstance(bias, tc.Tensor):
            bias_requires_grad = bias.requires_grad
        else:
            bias_requires_grad = False
        if bias is not None:
            bias = beta * bias
        else:
            bias = 0.0
        if margin is not None:
            margin = beta * margin
        else:
            margin = 0.0
        C1 = beta * C
#        N0 = input.get_len(ith_cls)
#        N1 = len(input) - N0
#        R = input.get_len_list()
#        M = len(R)
#        if balance_factor == 0.5:
#            R = [0.5/((M-1)*i) for i in R]
#            R[ith_cls] = 0.5/N0
#        else:
#            R = [(1.0-balance_factor)/N1]*M
#            R[ith_cls] = balance_factor/N0
        input.reiter()
#        cls_label = input.get_label(ith_cls)
        obj0 = 0.0
        obj1 = 0.0
        db = 0.0 if bias_requires_grad else None#tc.Tensor([0.0])
        dQ = 0.0
        for X, label in input:
            r = R[label]
#            ipdb.set_trace()
            if label == ith_cls:
                if r == 0.0:
                    continue
                s = 1.0
                f = X.mm(C1)
            else:
                if r == 0.0:
                    continue
                s = -1.0
                f = X.mm(C1.mul(-1.0))
            bias0 = s*bias - margin
            if bias0 is not 0.0:
                f.add_(bias0)
            f.sigmoid_()
            obj0 += -r*f.add(eps).log_().sum()
            f.mul_(-1.0).add_(1.0)
            dQ = dQ - s*r*beta*X.t().mm(f)
            if bias_requires_grad:
                db = db - s*r*beta*f.sum(0)
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
        argnum = tc.tensor([9])
        ctx.save_for_backward(dC, db, argnum)
#        ipdb.set_trace()
        return obj0 + obj1

