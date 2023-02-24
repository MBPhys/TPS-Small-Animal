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
    """
    Convert radians to degrees.

    Parameters:
        rad : float or array-like
           Angle(s) in radians to be converted to degrees.

    Returns:
        float or ndarray
           The corresponding angle(s) in degrees.

    Examples
    --------
    >>> radtoangle(np.pi/2)
    90.0
    >>> radtoangle([0, np.pi/4, np.pi/2])
    array([  0.,  45.,  90.])
    """
    angle=rad*180/ np.pi
    return angle

def angletorad(angle):
    """
    Convert angle from degrees to radians.

    Parameters:
        angle : float
           Angle in degrees.

    Returns:
        float
           Angle converted to radians.
    """
    rad= angle*np.pi/180
    return rad 

    
def check_np_elements(array):
    """
    Check if the input NumPy array has non-zero number of elements.

    Parameters:
        array (numpy.ndarray): The input array to check.

    Returns:
        bool: True if the input array has non-zero number of elements, False otherwise.

    Raises:
        ValueError: If the input is not a NumPy array.

    """
    return array.ndim and array.size

def resample_image_2D(data, original_spacing, target_spacing, order_data=3, cval_data=0):
    """
    Resamples a 2D image with the specified spacing.

    Paramters:
        data (ndarray): The input 2D image to be resampled.
        original_spacing (tuple): A tuple representing the original pixel spacing in (y, x) order.
        target_spacing (tuple): A tuple representing the target pixel spacing in (y, x) order.
        order_data (int): The order of the spline interpolation used for resampling. Default is 3.
        cval_data (float): The value to use for padding outside the boundary of the input image.
                           Default is 0.

    Returns:
        ndarray: The resampled 2D image with the same dtype as the input.

    Raises:
        AssertionError: If the input data is None or its shape is not 2D.

    """
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
    """
    Normalize the input array `x` to have the same mean-squared error (MSE)
    as the target array `target`.

    Parameters:
        x : np.ndarray
            The input array to be normalized.
        target : np.ndarray
                The target array to match the MSE of.

    Returns:
        np.ndarray
            The normalized input array `x` with the same MSE as `target`.

    Notes
    -----
    This function computes the scaling factor `alpha` and offset `beta` using
    the covariance between `x` and `target`, such that the resulting normalized
    array `alpha*x + beta` has the same MSE as `target`.

    Examples
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> target = np.array([3, 4, 5, 6])
    >>> normalize_minmse(x, target)
    array([2.5, 3. , 3.5, 4. ])

    """
    cov = np.cov(x.flatten(),target.flatten())
    alpha = cov[0,1] / (cov[0,0]+1e-10)
    beta = target.mean() - alpha*x.mean()
    return alpha*x + beta

def create_nonzero_mask(data, channel=False):
    """
    Create a binary mask from the input data that marks the non-zero voxels.

    Parameters:
        data (ndarray): Input data with shape (Z, Y, X) for 3D or (Y, X) for 2D.
                        Alternatively, it can have shape (C,Z,Y,X) for 3D with
                        multiple channels.
        channel (bool, optional): Set to True when `data` has multiple channels.
                                  Defaults to False.

    Returns:
        ndarray: A binary mask with the same shape as `data` that marks the non-zero voxels.
    """
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
    """Normalize the intensity values of a given image data.

    Parameters:
        data (np.ndarray): The 3D medical image data to be normalized.
        use_nonzero_mask (bool): Whether to use a mask that only includes non-zero voxels. Defaults to True.
        range_norm (bool): Whether to perform range normalization. Defaults to True.

    Returns:
        np.ndarray: The normalized image data.

    Raises:
        TypeError: If the input data is not a numpy array.
    """
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
    """
    Normalizes input data to a specified range.

    Parameters:
        data (ndarray): Input data with shape (C, Z, Y, X), shape (C, Y, X), shape (Z, Y, X) or shape (Y, X)
        rnge (tuple, optional): The desired range for the output data. Default is (0, 1).
        mask (ndarray, optional): A boolean mask indicating which values to normalize. Default is None.
        per_channel (bool, optional): If True, normalize each channel independently. Default is True.
        eps (float, optional): A small value to avoid dividing by zero. Default is 1e-8.

    Returns:
        ndarray: Normalized data with the same shape as the input data.
    """
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
    """
    Normalizes the input data using min-max normalization.

    Parameters:
        data (numpy.ndarray): Input data to be normalized.
        eps (float): A small value added to the denominator to avoid division by zero.

    Returns:
        numpy.ndarray: Normalized data with values between 0 and 1.
    """
    mn = data.min()
    mx = data.max()
    data_normalized = data - mn
    old_range = mx - mn + eps
    data_normalized /= old_range
    
    return data_normalized    

def tensor_exp2torch(T, BATCH_SIZE, device, data_type=torch.float, grad=True):
    """
    Converts a numpy array to a PyTorch tensor.

    Paramters:
        T (numpy.ndarray): Input tensor 
        BATCH_SIZE (int): Batch size to repeat the tensor.
        device (str): Device on which to create the tensor ('cpu' or 'cuda').
        data_type (torch.dtype, optional): Data type of the resulting tensor (default=torch.float).
        grad (bool, optional): Whether the tensor requires gradients (default=True).

    Returns:
        torch.Tensor: A PyTorch tensor with shape (BATCH_SIZE, Channel, Input tensor dimensions).

    Example:
        >>> T = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> tensor_exp2torch(T, 2, 'cuda')
        tensor([[[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]],

                [[1., 2., 3.],
                 [4., 5., 6.],
                 [7., 8., 9.]]], device='cuda:0', dtype=torch.float32, requires_grad=True)
    """
    T = np.expand_dims(T, axis=0)
    T = np.expand_dims(T, axis=0)
    T = np.repeat(T, BATCH_SIZE, axis=0)

    T = torch.tensor(T, dtype=data_type, requires_grad=grad, device=device)

    return T
    
def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, no_pad_left_side=False, shape_must_be_divisible_by=None):
"""
Pad an n-dimensional numpy array `image` to a desired `new_shape`.

Parameters
----------
image : ndarray
    The input array to be padded.
new_shape : tuple or ndarray, optional
    The desired new shape of the input array after padding. If not provided,
    `shape_must_be_divisible_by` must be set and the new shape will be the
    smallest possible shape that is divisible by the values in
    `shape_must_be_divisible_by`.
mode : str, optional
    One of the following string values indicating the type of padding to use:
    'constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum',
    'reflect', 'symmetric', 'wrap'.
kwargs : dict, optional
    Additional keyword arguments to be passed to the `np.pad` function.
return_slicer : bool, optional
    Whether to also return a tuple of slice objects representing the region
    of the padded array that corresponds to the original unpadded array.
no_pad_left_side : bool, optional
    Whether to avoid padding on the left side of each dimension. Default is False.
shape_must_be_divisible_by : tuple or ndarray, optional
    The values by which the new shape must be divisible. If not provided,
    `new_shape` must be set.

Returns
-------
padded_image : ndarray
    The input array padded to the desired shape.
slicer : tuple of slice objects, optional
    If `return_slicer` is True, a tuple of slice objects representing the
    region of the padded array that corresponds to the original unpadded array.

Notes
-----
The function pads the input array such that the difference between the old and
new shapes is evenly distributed across all dimensions. The `shape_must_be_divisible_by`
parameter is particularly useful when padding images for convolutional neural networks,
as it ensures that the dimensions of the image are divisible by a given factor.

References
----------
1. numpy.pad documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

2. Isensee Fabian, Jäger Paul, Wasserthal Jakob, Zimmerer David, Petersen Jens, Kohl Simon, 
Schock Justus, Klein Andre, Roß Tobias, Wirkert Sebastian, Neher Peter, Dinkelacker Stefan, 
Köhler Gregor, Maier-Hein Klaus (2020). batchgenerators - a python framework for data 
augmentation. doi:10.5281/zenodo.3632567
"""
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
    
