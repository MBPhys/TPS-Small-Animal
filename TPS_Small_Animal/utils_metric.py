#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 16:42:22 2022

@author: marc
"""

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss



class Gradient_NCCPRO(_Loss):
    """
    Computes the gradient-normalized cross correlation (NCC) loss between two tensors, with or without a mask.

    Parameters:
        use_mask (bool): If True, applies a mask to the input tensors.
        use_ncc (bool): If True, computes the NCC loss; if False, computes the Gradient-NCC loss.
    
    Attributes:
        None

    Methods:
        cal_ncc(I, J, eps): Computes the local sums and NCC loss between two tensors I and J.
        gradncc(I, J, device='cuda', win=None, eps=1e-10): Computes the gradient-NCC loss between two tensors I and J.
        ncc(I, J, device='cuda', win=None, eps=1e-10): Computes the NCC loss between two tensors I and J.

    Returns:
        A scalar tensor representing the NCC loss or gradient-NCC loss.

    Example:
        loss_fn = Gradient_NCCPRO(use_mask=True, use_ncc=False)
        loss = loss_fn.forward(I, J)
    """

  def __init__(self, use_mask: bool = False, use_ncc: bool =False):
      super().__init__()
      if use_ncc and use_mask:
         self.forward = self.ncc_mask
      elif use_ncc and not use_mask: 
         self.forward = self.ncc
      elif not use_ncc and use_mask:  
         self.forward = self.gradncc_mask 
      else:
         self.forward = self.gradncc    


  def cal_ncc(self, I, J, eps):
    # compute local sums via convolution
    cross = (I - torch.mean(I)) * (J - torch.mean(J))
    I_var = (I - torch.mean(I)) * (I - torch.mean(I))
    J_var = (J - torch.mean(J)) * (J - torch.mean(J))

    cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var*J_var + eps))

    #test = torch.mean(cc)
    return torch.mean(cc)

# Gradient-NCC Loss
  def gradncc(self, I, J, device='cuda', win=None, eps=1e-10):
        # compute filters
        with torch.no_grad():
            kernel_X = torch.Tensor([[[[1 ,0, -1],[2, 0 ,-2], [1, 0 ,-1]]]])
            kernel_X = torch.nn.Parameter( kernel_X, requires_grad = False )
            kernel_Y = torch.Tensor([[[[1, 2, 1],[0, 0, 0], [-1, -2 ,-1]]]])
            kernel_Y = torch.nn.Parameter( kernel_Y, requires_grad = False )
            SobelX = nn.Conv2d( 1, 1, 3, 1, 1, bias=False)
            SobelX.weight = kernel_X
            SobelY = nn.Conv2d( 1, 1, 3, 1, 1, bias=False)
            SobelY.weight = kernel_Y

            SobelX = SobelX.to(device)
            SobelY = SobelY.to(device)

        Ix = SobelX(I)
        Iy = SobelY(I)
        Jx = SobelX(J)
        Jy = SobelY(J)

        return  1-0.5*self.cal_ncc(Ix, Jx, eps)-0.5*self.cal_ncc(Iy, Jy, eps)

# NCC loss
  def ncc(self, I, J, device='cuda', win=None, eps=1e-10):  
    return 1-self.cal_ncc(I, J, eps)
