''' 
File: `util.py`
Description:
    Functions for computing Jacobians of arbitrary functions, thanks to the pytorch computational graph
'''

import torch
import torch.nn as nn

def jacobian(y, x):
    '''
    Jacobian of y with respect to x.

    Adapted from [Deepreach](https://github.com/smlbansal/deepreach/blob/master/diff_operators.py) `diff_operators.py`

    Bansal, Somil, and Claire J. Tomlin. "Deepreach: A deep learning approach to high-dimensional reachability."
    2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021.
    '''
    batch_size, num_observations = y.shape[:2]
    jac = torch.zeros(batch_size, num_observations, y.shape[-1], x.shape[-1], dtype=y.dtype).to(y.device)

    for i in range(y.shape[-1]):
        y_flat = y[...,i].view(-1, 1)
        jac[:, :, i, :] = torch.autograd.grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1
    return jac, status

def jacobian_f(y, x):
    '''
    Jacobian of y with respect to x. Single batch

    Adapted from [Deepreach](https://github.com/smlbansal/deepreach/blob/master/diff_operators.py) `diff_operators.py`

    Bansal, Somil, and Claire J. Tomlin. "Deepreach: A deep learning approach to high-dimensional reachability."
    2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021.
    '''
    num_observations = y.shape[0]
    jac = torch.zeros(num_observations, y.shape[-1], x.shape[-1], dtype=y.dtype).to(y.device)

    for i in range(y.shape[-1]):
        y_flat = y[...,i].view(-1, 1)
        jac[:, i, :] = torch.autograd.grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1
    return jac, status
