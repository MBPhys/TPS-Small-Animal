#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:42:52 2022

@author: marc
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas



def plot__regi_steps(fig, viewer, proj_mov, proj_init_numpy0, target,\
                   gradncc_sim_list, angle_0, angle_1, angle_2, trans_1, trans_2):
    """
    Plot the registration steps of 3D-2D registration

    Parameters:
    -----------
    fig : matplotlib figure
        Matplotlib figure object for plotting the registration steps
    viewer : napari viewer
        Napari viewer object for 3D visualization
    proj_mov : torch.Tensor
        Tensor of moving DRR projections
    proj_init_numpy0 : numpy.ndarray
        Initial DRR projection as numpy array
    target : torch.Tensor
        Tensor of target projections
    gradncc_sim_list : list
        List of gradient normalized cross-correlation similarity measure
    angle_0 : list
        List of rotation angles around z-axis
    angle_1 : list
        List of rotation angles around y-axis
    angle_2 : list
        List of rotation angles around x-axis
    trans_1 : list
        List of translation values along x-axis
    trans_2 : list
        List of translation values along y-axis

    Returns:
    --------
    canvas_fig : matplotlib FigureCanvas
        Figure canvas object for embedding in GUI application
    """
    gradncc_sim_list_np = np.array(gradncc_sim_list)
   
    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0)
    ax1.set_title('Initial DRR')

    ax2.imshow(proj_mov_numpy0)
    ax2.set_title('Moving DRR')

    ax3.imshow(target_numpy0)
    ax3.set_title('Target')

    ax4.plot(np.array(angle_0)[:, 0], color='darkred', marker='o')
    ax4.plot(np.array(angle_2)[:, 0], color='red', marker='o')
    ax4.plot(np.array(angle_1)[:, 0], color='lightsalmon', marker='o')
    ax4.set_title('Rotation degrees of freedom')
    ax4.set_ylabel('Radiant')
    ax4.set_xlabel('Iterations')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['rx', 'ry', 'rz'])
    ax4.grid()

    ax5.plot(np.array(trans_1)[:, 0] * target_numpy0.shape[0] , color='navy', marker='^')
    ax5.plot(np.array(trans_2)[:, 0] * target_numpy0.shape[1], color='blue', marker='^')
    ax5.set_title('Translation degrees of freedom')
    ax5.set_ylabel('micrometer')
    ax5.set_xlabel('Iterations')
    ax5.legend(['x', 'y', 'z'])
    ax5.tick_params(axis='y', labelcolor='tab:blue')
    ax5.grid()

    ax6.plot(gradncc_sim_list_np, marker='o')
    ax6.set_title('GradNCC Similarity')
    ax6.set_xlabel('Iterations')
    ax6.grid()
    
    canvas_fig=FigureCanvas(fig)
     
    return canvas_fig  

