# display.py
"""
Created on Wed May 23 20:19:56 2018

@author: Wentao Huang
"""
from __future__ import absolute_import, division, print_function
#import ipdb
import math
import torch as tc
from .helper import is_numpy

__all__ = [
    "make_grid",
    "make_grid_2d",
    "imshow",
    "show1d",
    "show1d2",
]

irange = range

def make_grid(tensor, nrow=None, padding=1,
              normalize=True, range=None, scale_each=False, pad_value=0.0):
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (tc.is_tensor(tensor) or
            (isinstance(tensor, list) and all(tc.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = tc.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = tc.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = tc.cat((tensor, tensor, tensor), 1)
#    ipdb.set_trace()    
    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"
        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)
    if tensor.size(0) == 1:
        return tensor.squeeze()
    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(nmaps)))
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-6)


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))


def make_grid_2d(tensor, shape=None, nrow=None, padding=1,
                 normalize=True, range=0, scale_each=False, pad_value=0.0):
    if not (tc.is_tensor(tensor) or
            (isinstance(tensor, list) and all(tc.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))
    assert tensor.dim() == 2
    N, L = tensor.size()
    if not isinstance(shape, tuple):
        if shape is None or shape == 2 or shape == 3:
            k = int(math.sqrt(L))
            if k*k == L and shape != 3:
                shape = (1, k, k)
            else:
                k = int(math.sqrt(L/3.0))
                if 3*k*k == L:
                    shape = (3, k, k)
                else:
                    raise ValueError('shape is None and has to be assigned by a 2 or 3 dimensional tuple')
    elif len(shape) == 2:
        shape = (1,) + shape
    elif len(shape) == 1 or len(shape) > 3:
        raise ValueError('shape must be a 2 or 3 dimensional tuple')
    if not (shape[0] == 1 or shape[0] == 3):
        raise ValueError('shape[0] must be equal to 1 or 3')
#    ipdb.set_trace()    
    if normalize:
        if range is not None and (not isinstance(range, tuple)):
            flag = True
            range = None
        else:
            flag = False
            if range is not None:
                assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified."
                " min and max are numbers"
        if not scale_each:
            if flag:
                max = tensor.abs().max()
                tensor = 0.5*(1.0 + tensor/(max+1e-6))
            else:
                norm_range(tensor, range)
    else:
        norm_range(tensor, None)
    images = []
    for n in irange(N):
        img = tensor[n,:]
        if normalize and scale_each:
            if flag:
                img = 0.5*(1.0 + img/(img.abs().max()+1e-6))
            else:
                norm_range(img, range)
        img = img.view(shape)
        if shape[0] == 1:
            img = tc.cat((img, img, img), 0)
        images.append(img)
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(N)))
#    ipdb.set_trace()
    images = make_grid(images, nrow, padding, normalize=False,
             range=None, scale_each=False, pad_value=pad_value)
    return images


def imshow(img, size=(8,8), issave=False, dpi=300, 
         fdir='figures', fname='filters', fmt='png', **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    if img is None:
        return False
    if tc.is_tensor(img):
        img = img.detach().numpy()
    if img.shape[0] == 3:
        img = np.transpose(img, (1,2,0))
    plt.figure(figsize=size)
    plt.imshow(img, interpolation='nearest', 
               origin='upper', aspect='equal', **kwargs)# cmap = plt.cm.gray, 
    plt.axis('off')
#    plt.title('Filters')
    if issave:
        from .helper import get_rootdir, path_join
        fn = path_join(get_rootdir(), fdir, fname + '.' + fmt)
        plt.savefig(fn, format=fmt, dpi=dpi)
    plt.show()
    return True
    
def show1d(data, size=(6,6), issave=False, dpi=300, 
            title=None, xlabel=None, ylabel=None,
            fdir='figures', fname='data', fmt='png', 
            *args, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    if data is None:
        return False
    if tc.is_tensor(data):
        data = data.detach().numpy().flatten()
    elif is_numpy(data):
        data = data.flatten()
    L = data.size
    if (L == 0) :
        return False
    x = np.linspace(1, L, L)
    plt.figure(figsize = size)#fname, 
    plt.plot(x, data, *args, **kwargs)#-o, markersize=3
    plt.xlim([0, L])
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    
    if issave:
        from .helper import get_rootdir, path_join
        fn = path_join(get_rootdir(), fdir, fname + '.' + fmt)
        plt.savefig(fn, format=fmt, dpi=dpi)
    plt.show()
    return True


def show1d2(data, nrow=None, size=(6,6), issave=False, dpi=300, 
             title=None, xlabel=None, ylabel=None,
             fdir='figures', fname='data', fmt='png', 
             *args, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    if data is None:
        return False
    data = data.detach().numpy()
    dim = data.ndim
    if dim == 1:
        return show1d(data, size, issave, dpi, title, xlabel, ylabel,
                      fdir, fname, fmt, *args, **kwargs)
    elif dim < 1 or dim > 2:
        return False

    N, L = data.shape
    x = np.linspace(1, L, L)
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(N)))
    ncol = int(math.ceil(N/nrow))
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=size,
                             sharex=True, sharey=True, squeeze=False)
    n = 0
    for i in range(nrow):
        if n >= N:
             break
        for j in range(ncol):
            if n >= N:
                break
            axes[i, j].plot(x, data[n,:], *args, **kwargs)#-o, markersize=3
            n += 1
            if j == 0 and ylabel is not None:
                axes[i, j].set_ylabel(ylabel)
            if i == nrow-1 and xlabel is not None:
                axes[i, j].set_xlabel(xlabel)
    plt.xlim([0, L])
    if title is not None:
        plt.suptitle(title)

    if issave:
        from .helper import get_rootdir, path_join
        fn = path_join(get_rootdir(), fdir, fname + '.' + fmt)
        plt.savefig(fn, format=fmt, dpi=dpi)
    plt.show()
    return True
