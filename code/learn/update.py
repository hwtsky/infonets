# update.py
"""
Created on Fri May 25 17:03:52 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division, print_function
#import ipdb
import time
import torch as tc
from .adsgd import AdSGD
from ..utils.data import OrderedDataset
from ..utils.helper import to_Param, copy_params, get_keys_vals


def update(object, cls, input, C, bias=None, 
           params=None, *args, **kwargs):
    F = getattr(object, cls, None)
    assert isinstance(input, OrderedDataset) and input.dim()==2
    assert isinstance(C, tc.Tensor) and C.requires_grad and C.dim()==2
    p = copy_params(is_raised=False, params=params)
    keyslist = ['max_epoches', 'beta', 'bias_requires_grad', 'lr', 'minlr',
                'tao', 'momentum_decay', 'isnorm', 'isrow', 'isorth', 'orthNum', 
                'save_history', 'display', 'eps', 'weight_end', 'bias_start']
    assignlist = [200, 1.0, False, 0.4, 1e-4, 0.9, 0.9, True, False, 
                  True, 50, True, 50, 1e-6, None, None]
    [max_epoches, beta, bias_requires_grad, lr, minlr,tao, momentum_decay, 
     isnorm, isrow, isorth, orthNum, save_history, display, eps,
      weight_end, bias_start] = get_keys_vals(keyslist, assignlist, params=p)
    if orthNum is None:
        orthNum = max_epoches
    if display is None:
        display = max_epoches
    if bias_start is None:
        bias_start = orthNum
    if bias_requires_grad and bias_start < 1:
        update_bias = True
    else:
        update_bias = False
    if weight_end is None:
        weight_end = max_epoches
    if not bias_requires_grad:
        weight_end = max_epoches
    if weight_end > 0:
        update_weight = True
    else:
        update_weight = False
    if bias_requires_grad:
        if bias is None:
            KA = C.size(1)
            bias = tc.zeros(KA)
        bias = to_Param(bias, True)
        optim = AdSGD([C, bias], lr, minlr, momentum_decay,
                      tao, isorth, isrow, isnorm, eps, update_weight, update_bias)
    else:
        bias = to_Param(bias, False)
        optim = AdSGD([C], lr, minlr, momentum_decay,
                      tao, isorth, isrow, isnorm, eps, update_weight, update_bias)
    if bias_requires_grad:
        assert isinstance(bias, tc.Tensor) and bias.requires_grad
#    ipdb.set_trace()
    cls_labels = input.cls_labels
    input.cls_labels = list(range(len(cls_labels)))
    history = None
    Y0 = None
    Y1 = None
    start = time.clock()
    for t in range(1, 1+max_epoches):
        if isorth and t == orthNum + 1:
            isorth = False
            optim.param_groups[0]['isorth'] = isorth
        if t > weight_end:
            optim.param_groups[0]['update_weight'] = False
        else:
            optim.param_groups[0]['update_weight'] = True
        if bias_requires_grad:
            if t > bias_start:
#                ipdb.set_trace()
                optim.param_groups[0]['update_bias'] = True
            else:
                optim.param_groups[0]['update_bias'] = False
        optim.zero_grad()
        Y = F.apply(input, C, bias, beta, isorth, eps, *args, **kwargs)
        decay = False if Y0 is None else bool(Y>=Y0)
        Y.backward()
        optim.step(decay=decay)
        Y0 = Y
        if save_history:
            if history is None:
                history = Y.data
                if history.dim() == 0:
                    history = history.unsqueeze(0)
            else:
                d = Y.data
                if d.dim() == 0:
                    d = d.unsqueeze(0)
                history = tc.cat([history, d])
        if ((display > 0) and ((display == 1) or (t == 1) or
                              ((t%display) == 0) or (t == max_epoches))):
            if Y1 is None:
                Y1 = Y
            if 'lr_0' in optim.state:
                lrw = optim.state['lr_0']['lrt']
                lrw = 'lrw={:.2e}, '.format(lrw)
            else:
                lrw = ''
            if 'lr_1' in optim.state:
                lrb = optim.state['lr_1']['lrt']
                lrb = 'lrb={:.2e}, '.format(lrb)
            else:
                lrb = ''
            if bias_requires_grad:
                b = 'bias_mean={:.2e}, '.format(bias.mean().item())
            else:
                b = ''
            if t == 1:
                K, KA = C.size()
                print('\nRuning {}: max_epoches={}, in_size={}, '
                      'out_size={}'.format(cls, max_epoches, K, KA))
            elapsed = (time.clock()-start)/60.0
            print('[{}]: obj={:.2e}, Dobj={:.2e}, '.format(
                    t, Y.item(), (Y-Y1).item()) + b +
                  '{}{}etime={:.2e}m'.format(
                    lrw, lrb, elapsed))
            Y1 = Y
    input.reiter()
    input.cls_labels = cls_labels
    return C, bias, history


def update1(object, cls, input, C, bias=None, 
            params=None, *args, **kwargs):
    F = getattr(object, cls, None)
    assert isinstance(input, OrderedDataset) and input.dim()==2
    assert isinstance(C, tc.Tensor) and C.requires_grad and C.dim()==2
    p = copy_params(is_raised=False, params=params)
    keyslist = ['max_epoches', 'beta', 'bias_requires_grad', 'lr', 'minlr',
                'tao', 'momentum_decay', 'isnorm', 'isrow', 'isorth', 'orthNum', 
                'save_history', 'display', 'eps', 'weight_end', 'bias_start']
    assignlist = [200, 1.0, False, 0.4, 1e-4, 0.9, 0.9, True, False, 
                  True, 50, True, 50, 1e-6, None, None]
    [max_epoches, beta, bias_requires_grad, lr, minlr,tao, momentum_decay, 
     isnorm, isrow, isorth, orthNum, save_history, display, eps,
      weight_end, bias_start] = get_keys_vals(keyslist, assignlist, params=p)
    if orthNum is None:
        orthNum = max_epoches
    if display is None:
        display = max_epoches
    if bias_start is None:
        bias_start = orthNum
    if bias_requires_grad and bias_start < 1:
        update_bias = True
    else:
        update_bias = False
    if weight_end is None:
        weight_end = max_epoches
    if not bias_requires_grad:
        weight_end = max_epoches
    if weight_end > 0:
        update_weight = True
    else:
        update_weight = False
    if bias_requires_grad:
        if bias is None:
            KA = C.size(1)
            bias = tc.zeros(KA)
        bias = to_Param(bias, True)
        optim = AdSGD([C, bias], lr, minlr, momentum_decay,
                      tao, isorth, isrow, isnorm, eps, update_weight, update_bias)
    else:
        bias = to_Param(bias, False)
        optim = AdSGD([C], lr, minlr, momentum_decay,
                      tao, isorth, isrow, isnorm, eps, update_weight, update_bias)
    if bias_requires_grad:
        assert isinstance(bias, tc.Tensor) and bias.requires_grad
#    ipdb.set_trace()
    cls_labels = input.cls_labels
    input.cls_labels = list(range(len(cls_labels)))
    history = None
    Y0 = None
    Y1 = None
    decay = False
    start = time.clock()
    for t in range(1, 1+max_epoches):
        if isorth and t == orthNum + 1:
            isorth = False
            optim.param_groups[0]['isorth'] = isorth
        if t > weight_end:
            optim.param_groups[0]['update_weight'] = False
        else:
            optim.param_groups[0]['update_weight'] = True
        if bias_requires_grad:
            if t > bias_start:
                optim.param_groups[0]['update_bias'] = True
            else:
                optim.param_groups[0]['update_bias'] = False
        N = len(input)
        Y = 0.0
        input.reiter()
        for X, label in input:
            X = OrderedDataset([X])
            X.cls_labels[0] = label
            optim.zero_grad()
            y = F.apply(X, C, bias, beta, isorth, eps, *args, **kwargs)
            y.backward()
            optim.step(decay=decay)
            decay = False
            Y = Y + (1.0*X.size(0)/N)*y.item()
#            print('t={}, label={}, Y={}, y={}'.format(t, label, Y, y.item()))
#        ipdb.set_trace()
        decay = False if Y0 is None else bool(Y>=Y0)
        Y0 = Y
        if save_history:
            if history is None:
                history = tc.tensor([Y])
            else:    
                history = tc.cat([history, tc.tensor([Y])])
        if ((display > 0) and ((display == 1) or (t == 1) or
                              ((t%display) == 0) or (t == max_epoches))):
            if Y1 is None:
                Y1 = Y
            if 'lr_0' in optim.state:
                lrw = optim.state['lr_0']['lrt']
                lrw = 'lrw={:.2e}, '.format(lrw)
            else:
                lrw = ''
            if 'lr_1' in optim.state:
                lrb = optim.state['lr_1']['lrt']
                lrb = 'lrb={:.2e}, '.format(lrb)
            else:
                lrb = ''
            if bias_requires_grad:
                b = 'bias_mean={:.2e}, '.format(bias.mean().item())
            else:
                b = ''
            if t == 1:
                K, KA = C.size()
                print('\nRuning {}: max_epoches={}, in_size={}, '
                      'out_size={}'.format(cls, max_epoches, K, KA))
            elapsed = (time.clock()-start)/60.0
            print('[{}]: obj={:.2e}, Dobj={:.2e}, '.format(
                    t, Y, (Y-Y1)) + b +
                  '{}{}etime={:.2e}m'.format(
                    lrw, lrb, elapsed))
            Y1 = Y
    input.reiter()
    input.cls_labels = cls_labels
    return C, bias, history
