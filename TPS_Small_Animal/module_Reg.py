#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:18:12 2022
@author: marc
"""

import torch
import torch.nn as nn
from interpol.autograd import GridPull
from interpol import affine_grid

def euler2mat(angles):
    """
    Converts Euler angles to rotation matrix.

    Parameters:
        angles (torch.Tensor): Tensor of shape (B, 3), where B is the batch size and each row contains Euler angles
                               (in radians) for each axis in the order of Z, Y, and X.

    Returns:
        rotMat (torch.Tensor): Tensor of shape (B, 3, 3) representing the rotation matrix.

    Examples:
        >>> angles = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        >>> euler2mat(angles)
        tensor([[[ 0.9363, -0.2752,  0.2184],
                 [ 0.2896,  0.9564, -0.0366],
                 [-0.1987,  0.0978,  0.9752]],

                [[ 0.8253, -0.4096,  0.3880],
                 [ 0.5155,  0.8509, -0.1060],
                 [-0.2290,  0.3318,  0.9156]]])
    """
    B = angles.size(0)
    z, y, x = angles[:,0], angles[:,1], angles[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = zmat @ ymat @ xmat
    return rotMat


def matrix_for_translation(trans , device):
    """
    Constructs a 4x4 transformation matrix for translation.

    Parameters:
        trans (torch.Tensor): Tensor of size (B, 3) containing the translation vector.
        device (torch.device): Device on which the tensor should be constructed.

    Returns:
        torch.Tensor: Tensor of size (B, 4, 4) representing the 4x4 transformation matrix for translation.

    """
    B=trans.size(0)
    zeros = trans[:,0]*0
    ones = zeros+1
    identity_matrix=torch.stack([ones, zeros, zeros,
                        zeros, ones, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)
    trans_exp=trans.unsqueeze(axis=2)
    torch_rigid_matrix_3x3=torch.cat((identity_matrix,  trans_exp), axis=-1)
    torch_rigid_matrix_4x4=torch.cat((torch_rigid_matrix_3x3, torch.tensor([[[0.0,0.0,0.0,1.0]]], device=device)), axis=1)                              
    return torch_rigid_matrix_4x4

def translate_transform(x, torch_rigid_matrix_4x4):
    """
    Applies a 3D translation transform to a tensor using a given 4x4 transformation matrix.

    Parameters:
        x (torch.Tensor): The tensor to transform.
        torch_rigid_matrix_4x4 (torch.Tensor): The 4x4 rigid transformation matrix.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    grid_size=x.size()
    grid = affine_grid(torch_rigid_matrix_4x4, grid_size[2:])
    transform=GridPull.apply
    x = transform(x, grid, 1,  'zeros',1)
    return x
    
    
   
class Reg_model(nn.Module):
        """
          This class defines the registration model that performs rigid transformation on the input tensor.

        Parameters:
          None

        Returns:
          None
        """
        def __init__(self):
            super(Reg_model, self).__init__()
        
        def forward(self, x, y, rot_0, rot_1, rot_2, trans_1, trans_2, center_point):
        """
        This function applies rigid transformation on the input tensor 'x' based on the given rotation and translation parameters.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, D, H, W] where B is the batch size, C is the number of channels, D, H, W are the spatial dimensions.
            y (torch.Tensor): Ground truth tensor of the same shape as 'x'.
            rot_0 (torch.Tensor): Tensor of shape [B,] containing rotation angles along the x-axis.
            rot_1 (torch.Tensor): Tensor of shape [B,] containing rotation angles along the y-axis.
            rot_2 (torch.Tensor): Tensor of shape [B,] containing rotation angles along the z-axis.
            trans_1 (torch.Tensor): Tensor of shape [B,] containing translation values along the x-axis.
            trans_2 (torch.Tensor): Tensor of shape [B,] containing translation values along the y-axis.
            center_point (torch.Tensor): Tensor of shape [B, 3] containing the center points of rotation.

        Returns:
            x_2d (torch.Tensor): Tensor of shape [B, C, H, W] containing the transformed 2D tensor after applying the rigid transformation on 'x'.
            x (torch.Tensor): Tensor of the same shape as 'x_2d' containing the transformed 3D tensor after applying the rigid transformation on 'x'.
            y (torch.Tensor): Ground truth tensor of the same shape as 'x_2d'.
        """
            grid_size=x.size()
            trans_matrix_1=matrix_for_translation(-center_point, torch.device("cuda"))
            trans_matrix_2=matrix_for_translation(center_point, torch.device("cuda"))
            
            
            
            
            trans_torch=torch.cat((trans_1.detach()*0, trans_1*x.size()[3], trans_2*x.size()[4]),     axis=1)
            angles_torch=torch.cat((rot_0, rot_1, rot_2),     axis=1)     
            rotation_matrix=euler2mat(angles_torch)
            
            trans_torch_exp=trans_torch.unsqueeze(axis=2)
            torch_rigid_matrix_3x3=torch.cat((rotation_matrix,  trans_torch_exp), axis=-1)
            torch_rigid_matrix_4x4=torch.cat((torch_rigid_matrix_3x3, torch.tensor([[[0.0,0.0,0.0,1.0]]], device=torch.device("cuda"))), axis=1)                              
            
            torch_rigid_matrix_4x4_center=torch.matmul(trans_matrix_2,torch.matmul(torch_rigid_matrix_4x4, trans_matrix_1))
            grid = affine_grid(torch_rigid_matrix_4x4_center, grid_size[2:])
            transform=GridPull.apply
            x = transform(x, grid, 1,  'zeros',1)
            x_2d=torch.mean(x, dim=2)
            return x_2d, x, y
