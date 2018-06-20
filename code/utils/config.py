# config.py
"""
Created on Fri Jun 15 01:49:06 2018

@author: Wentao Huang
"""
from __future__ import print_function, unicode_literals
import warnings
from collections import OrderedDict
from .helper import copy_params

class Config(object):
    
    def __init__(self, params=None, **kwargs):
        p = copy_params(params=params, **kwargs)
        for k, v in p.items():
            setattr(self, k, v)
    
    def parse(self, isprint=False, params=None, **kwargs):
        p = copy_params(params=params, **kwargs)
        for k, v in p.items():
            if not hasattr(self, k):
                warnings.warn('Warning: opt has not attribut {}'.format(k))
            setattr(self, k, v)
        if isprint:
            self.print_config()
    
    def to_dict(self):
        D = OrderedDict()
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                D[k] = getattr(self, k)
        return D
            
    def print_config(self, prefix=None):
        if prefix is None:
            prefix = 'User config'
        print(prefix + ': ')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k+' = ', getattr(self, k))
    
    def _get_name(self):
        return self.__class__.__name__
    
    def __repr__(self):
        main_str = self._get_name() + '(\n'
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                main_str += '    {} = {}\n'.format(k, getattr(self, k))
        main_str +=')'
        return main_str

