''' 
File: `dpi_edmdc.py`
Description:
    Delay coordinate physics informed extended dynamic mode decomposition with control.
    Uses a PDE prior to acquire a more accurate model.

    This form does not EXPLICITLY use the Strang splitting form, 
        it uses a specific form that has K_t^h ~= (I + K_t^h),
        which tends to work slightly better when terms of h are known
        to be 0
'''

import numpy as np
import torch
from scipy.linalg import expm

from util import jacobian_f

class DPI_EDMDc:
    '''
    Delay coordinate physics informed extended dynamic mode decomposition with control
    '''
    def __init__(self, x_dict_obj, xu_dict_obj, pde_obj, dt, option=1):
        '''
        Initialize DPI_EDMDc model
        
        Args:
            x_dict_obj (obj): Dictionary object for feature generation for state
            xu_dict_obj (obj): Dictionary object for feature generation for state and control
            pde_obj (obj): PDE object for physics-informed learning
            dt (float): Time step
            option (int): Option for computing the physics informed model. 1 for continuous-time, 2 for discrete-time
        '''
        self.x_dict_obj = x_dict_obj
        self.xu_dict_obj = xu_dict_obj
        self.pde_obj = pde_obj
        self.dt = dt
        self.option = option
    
    def fit(self, X, U, Y):
        '''
        Fit DPI_EDMDc model
        
        X should be in the Hankel matrix form
        ```
        | x_k      x_{k+1}  ... |
        | x_{k-1}  x_k      ... |
        | ...      ...          |
        ```
        
        Args:
            X (torch.tensor): State data matrix, in shape [N, D_STATE * DELAY]
            U (torch.tensor): Input data matrix, in shape [N, D_INPUT]
            Y (torch.tensor): State data matrix, in shape [N, D_STATE]
        '''
        _, D_STATE = Y.shape
        DELAY = X.shape[1] // D_STATE
        
        self.D_STATE = D_STATE
        self.DELAY = DELAY
        
        #====================#
        # Setup Dictionaries #
        #====================#
        self.x_dict_obj.fit(X[:,:D_STATE])
        if self.xu_dict_obj is not None:
            self.xu_dict_obj.fit(torch.cat([X, U], dim=1))
        
        Z = self._compute_Z(X, D_STATE, DELAY)
        ZP = self.x_dict_obj.transform(Y)
        if self.xu_dict_obj is not None:
            ZU = self.xu_dict_obj.transform(torch.cat([X, U], dim=1))
        else:
            ZU = U
        
        #========================#
        # Compute Model from PDE #
        #========================#
        if self.option == 1:
            # OPTION 1: Continuous-time #
            Z_ = Z.detach().cpu().numpy()
            
            F = self.pde_obj(X[:,:D_STATE])[:,:,None].detach().cpu().numpy()
            J, _ = jacobian_f(Z, X[:,:D_STATE])
            J = J.detach().cpu().numpy()
            JF = np.squeeze(J @ F, axis=2)
            
            A = np.linalg.lstsq(Z_, JF, rcond=None)[0].T
            exp_A = expm(self.dt * A)
            self.L1 = exp_A
        else:
            # OPTION 2: Discrete-time #
            Z_ = Z.detach().cpu().numpy()
            
            F = self.pde_obj(X[:,:D_STATE])
            
            # Create fake time-advanced matrix from F
            # Use midpoint method
            temp = X[:,:D_STATE] + self.dt/2 * F
            F = self.pde_obj(temp)
            X_prime = X[:,:D_STATE] + self.dt * F
            
            Z_prime = self.x_dict_obj.transform(X_prime)
            Z_prime_ = Z_prime.detach().cpu().numpy()
            
            exp_A = np.linalg.lstsq(Z_, Z_prime_, rcond=None)[0].T
            self.L1 = exp_A
        
        #======================================#
        # Compute Discrete Perturbation Matrix #
        #======================================#
        ZP_ = ZP.detach().cpu().numpy()
        diff = ZP_ - Z_ @ self.L1.T
        self.L2 = np.linalg.lstsq(ZU, diff, rcond=None)[0].T
    
    def predict(self, x, U):
        '''
        Predict next state
        
        x should be in the Hankel matrix form
        ```
        | x_k    |
        | x_{k-1}|
        | ...    |
        ```
        
        Args:
            x (torch.tensor): Initial data state [D_STATE*DELAY]
            U (torch.tensor): Input data sequence, in shape [N', D_INPUT]
        '''
        D_STATE = self.D_STATE
        DELAY = self.DELAY
        
        x_res = np.zeros((U.shape[0]+1, D_STATE))
        x_res[0,:] = x[:D_STATE]
        
        z = self._compute_Z(x[None,:], D_STATE, DELAY).detach().numpy()
        if self.xu_dict_obj is not None:
            zu= self.xu_dict_obj.transform(torch.cat([x, U[0,:]])[None,:]).detach().numpy()
        else:
            zu= U[0,:][None,:].detach().numpy()
        
        del_temp = x.clone().detach()
        
        for i in range(U.shape[0]):
            zp = self.L1 @ z.T + self.L2 @ zu.T
            x_res[i+1,:] = self.x_dict_obj.untransform(zp.T)
            
            del_temp[D_STATE:] = del_temp[:-D_STATE]
            del_temp[:D_STATE] = torch.tensor(x_res[i+1,:], dtype=torch.float64)
            
            z = self._compute_Z(del_temp[None,:], D_STATE, DELAY).detach().numpy()
            
            if self.xu_dict_obj is not None:
                zu= self.xu_dict_obj.transform(torch.cat([del_temp, U[i,:]])[None,:]).detach().numpy()
            else:
                zu= U[i,:][None,:].detach().numpy()

        # Second return is compat return
        return x_res, None
    
    def _compute_Z(self, X, D_STATE, DELAY):
        '''
        Compute state-dependent Z matrix
        
        Args:
            X (torch.tensor): State data matrix, in shape [N, D_STATE * DELAY]
            D_STATE (int): State dimension
            DELAY (int): Delay dimension
        '''
        Z = []
        
        for i in range(DELAY):
            Z.append(self.x_dict_obj.transform(X[:,i*D_STATE:(i+1)*D_STATE]))
        Z = torch.cat(Z, dim=1)
        
        return Z
        
        
        

