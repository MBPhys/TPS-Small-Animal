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
    B=trans.size(0)
    zeros = trans[:,0]*0
    #print(zeros)
    ones = zeros+1
    identity_matrix=torch.stack([ones, zeros, zeros,
                        zeros, ones, zeros,
                        zeros, zeros, ones], dim=1).reshape(B, 3, 3)
    trans_exp=trans.unsqueeze(axis=2)
    torch_rigid_matrix_3x3=torch.cat((identity_matrix,  trans_exp), axis=-1)
    torch_rigid_matrix_4x4=torch.cat((torch_rigid_matrix_3x3, torch.tensor([[[0.0,0.0,0.0,1.0]]], device=device)), axis=1)                              
    return torch_rigid_matrix_4x4

def translate_transform(x, torch_rigid_matrix_4x4):
    grid_size=x.size()
    grid = affine_grid(torch_rigid_matrix_4x4, grid_size[2:])
    transform=GridPull.apply
    x = transform(x, grid, 1,  'zeros',1)
    return x
    
    
   
class Reg_model(nn.Module):
        def __init__(self):
            super(Reg_model, self).__init__()
        
        def forward(self, x, y, rot_0, rot_1, rot_2, trans_1, trans_2, center_point):
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