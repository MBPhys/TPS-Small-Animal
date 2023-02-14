#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:32:34 2021

@author: Marc Boucsein
"""
import numpy as np 
from scipy.ndimage import binary_fill_holes
from skimage.util import img_as_float32
from skimage.transform import resize
import torch

def radtoangle(rad):
    angle=rad*180/ np.pi
    return angle

def angletorad(angle):
    rad= angle*np.pi/180
    return rad 

    
def check_np_elements(array):
    return array.ndim and array.size

def resample_image_2D(data, original_spacing, target_spacing, order_data=3, cval_data=0):
    assert not (data is None)
    if data is not None:
        assert len(data.shape)==2, "data must be y x"
        shape = np.array(data.shape)
    else:
        print('data is None')
    new_shape = np.round(((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)).astype(int)
    dtype_data = data.dtype
    new_shape = np.array(new_shape)
    if data is not None:
       if np.any(shape != new_shape):
          data = data.astype(float)
          reshaped_final_data=resize(data, new_shape, order_data, mode="edge",  cval=cval_data, clip=True, anti_aliasing=False)
          del data
          return reshaped_final_data.astype(dtype_data) 
       else:
          print("no resampling necessary")
          return data   


def normalize_minmse(x, target):
    cov = np.cov(x.flatten(),target.flatten())
    alpha = cov[0,1] / (cov[0,0]+1e-10)
    beta = target.mean() - alpha*x.mean()
    return alpha*x + beta

def create_nonzero_mask(data, channel=False):
    assert len(data.shape) == 4 or len(data.shape) == 3 or len(data.shape) == 2, "data must have shape (Z, Y, X), shape (Y, X) or shape (C,Z,Y,X)"

    if channel:
       nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
       for c in range(data.shape[0]):
            this_mask = data[c] != 0
            nonzero_mask = nonzero_mask | this_mask
           
    else:  
        nonzero_mask = np.zeros(data.shape, dtype=bool)

        this_mask = data != 0
        nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    del this_mask
    return nonzero_mask

def intensity_normalize_img(data, use_nonzero_mask=True, range_norm=True):
        print("normalization...")
        data=img_as_float32(data)
        nonzero_mask = create_nonzero_mask(data, channel=False)
        nonzero_mask = nonzero_mask.astype(int)            

        mask = np.ones(nonzero_mask.shape, dtype=bool)
                    
             
        data[mask] = (data[mask] - data[mask].mean()) / (data[mask].std() + 1e-8)
        data[mask == 0] = 0
        if range_norm:
            data=range_normalization(data, mask=mask, per_channel=False)
                #masks[c]=mask
               
        del nonzero_mask 
        del mask

        print("normalization done")

        return data #, mask_da  
    
def range_normalization(data, rnge=(0, 1), mask=None, per_channel=True, eps=1e-8):
    assert len(data.shape) == 4 or len(data.shape) == 3 or len(data.shape) == 2, "data must have shape (C, Z, Y, X), shape (C, Y, X), shape (Z, Y, X) or shape (Y, X)"
    data_normalized = np.zeros(data.shape, dtype=data.dtype)    
    if per_channel:
            for c in range(data.shape[0]):
                if mask is not None:
                   data_normalized[c][mask] = min_max_normalization(data[c][mask], eps)
                else:
                   data_normalized[c] = min_max_normalization(data[c], eps) 
    else:
        if mask is not None:  
                data_normalized[mask]= min_max_normalization(data[mask], eps)
        else:
                data_normalized= min_max_normalization(data, eps)

    data_normalized *= (rnge[1] - rnge[0])
    data_normalized += rnge[0]
    del data
    return data_normalized


def min_max_normalization(data, eps):
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range
    
    return data_normalized    

def tensor_exp2torch(T, BATCH_SIZE, device, data_type=torch.float, grad=True):
   # print(data_type)
    T = np.expand_dims(T, axis=0)
    T = np.expand_dims(T, axis=0)
    T = np.repeat(T, BATCH_SIZE, axis=0)

    T = torch.tensor(T, dtype=data_type, requires_grad=grad, device=device)

    return T
    
def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, no_pad_left_side=False, shape_must_be_divisible_by=None):
    if kwargs is None:
        kwargs = {'constant_values': 0}

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = len(image.shape) - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    if not isinstance(new_shape, np.ndarray):
        new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i] % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    if no_pad_left_side:
       pad_below=difference *0
       pad_above=difference + difference % 2
    else:      
       pad_below = difference // 2
       pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = list(slice(*i) for i in pad_list)
        return res, slicer 
    
