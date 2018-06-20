# adsgd.py
"""
Created on Sun May 20 23:28:32 2018

@author: Wentao Huang
"""
import ipdb
from collections import defaultdict
import torch as tc
from torch.optim import Optimizer
from ..utils.methods import Gram_Schmidt_Orth

required = object()


class Optim(Optimizer):
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Arguments:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults
        if isinstance(params, tc.Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                            tc.typename(params))
        self.state = defaultdict(dict)
        self.param_groups = []
#        param_groups = list(params)
        if not isinstance(params, list):
            param_groups = [params]
        else:
            param_groups = params
#        ipdb.set_trace()
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)


class AdSGD(Optim):

    def __init__(self, params, lr=4e-1, minlr=1e-4, momentum_decay=0.9,  
                 tao=0.9, isorth=True, isrow=False, isnorm=True, 
                 eps=1e-6, update_weight=True, update_bias=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum_decay:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))
        defaults = dict(lr=lr, minlr=minlr, momentum_decay=momentum_decay,
                        tao=tao, isorth=isorth, isrow=isrow, isnorm=isnorm, eps=eps, 
                        update_weight=update_weight, update_bias=update_bias)
        super(AdSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('isorth', True)
            group.setdefault('isnorm', True)
            
    def step(self, closure=None, decay=False):#
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            minlr = group['minlr']
            momentum_decay = group['momentum_decay']
            tao = group['tao']
            eps = group['eps']
            isorth = group['isorth']
            isrow = group['isrow']
            isnorm = group['isnorm']
            update_weight = group['update_weight']
            update_bias = group['update_bias']

            pname = -1
            for p in group['params']:
                pname += 1
                if p is None or p.grad is None:
                    continue
                dim = p.data.dim()
                d_p = p.grad.data
                
                param_lr = self.state['lr_'+str(pname)]
                if 'lrt' not in param_lr:
                    lrt = param_lr['lrt'] = lr
                else:
                    lrt = param_lr['lrt']
                    
                if dim <= 1:
                    if not update_bias:
                        continue
                    if d_p.dim() == 2:
                        d_p = d_p.squeeze(0)
                    dr = tc.mean((tc.abs(d_p)+eps)/(0.01+eps+tc.abs(p.data)))
                else:
                    if not update_weight:
                        continue
                    dc2 = tc.sqrt(tc.sum(d_p*d_p,0)) + eps
                    c2 = tc.sqrt(tc.sum(p.data*p.data,0)) + eps
                    dr = tc.mean(dc2/c2)
                d_p = d_p/dr
                
                if decay:
                    lrt = tao*lrt
                    lrt = lrt if lrt > minlr else minlr
                    param_lr['lrt'] = lrt
                
                if dim == 1 and d_p.dim() == 2:
                    d_p = d_p.squeeze(0)

                param_state = self.state[str(pname)]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = tc.zeros_like(p.data)
                else:
                    buf = param_state['momentum_buffer']
#                ipdb.set_trace()
                if momentum_decay == 0.0:
                    buf = d_p
                else:
                    buf.mul_(momentum_decay).add_(d_p.mul_(1.0-momentum_decay))
                p.data.add_(-lrt, buf)
                
                if dim == 2:
                    K, KA = p.data.size()
                    d = 1 if isrow else 0
                    if isorth:
                        if KA == 1 or K == 1:
                            p.data.div_(p.data.norm(2,dim=d,keepdim=True)+eps)
                        else:
                            p.data = Gram_Schmidt_Orth(p.data, isrow, eps)
                    elif isnorm:
                        p.data.div_(p.data.norm(2,dim=d,keepdim=True)+eps)
        return loss