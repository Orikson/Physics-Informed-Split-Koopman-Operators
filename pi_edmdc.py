''' 
File: `pi_edmdc.py`
Description:
    Physics informed extended dynamic mode decomposition with control.
    Uses a PDE prior to acquire a more accurate model.

    This form does not EXPLICITLY use the Strang splitting form, 
        it uses a specific form that has K_t^h ~= (I + K_t^h),
        which tends to work slightly better when terms of h are known
        to be 0
'''

import numpy as np
import torch
from util import jacobian_f
from scipy.linalg import expm

class PI_EDMDc:
    '''
    Physics informed extended dynamic mode decomposition with control
    '''
    
    def __init__(self, x_dict_obj, xu_dict_obj, pde_obj, dt, option=1):
        '''
        Initialize PI_EDMDc model
        
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
    
    def fit(self, X, U, Y=None, k=None, X2=None):
        '''
        Fit PI_EDMDc model
        
        Args:
            X (torch.tensor): State data matrix, in shape [N, D_STATE]
            U (torch.tensor): Input data matrix, in shape [N, D_INPUT]
            Y (torch.tensor): State data matrix, in shape [N, D_STATE]. If None, set Y to timeshifted X
            k (int): Number of principal components to use. If None, then no PCA is conducted
        '''
        self.k = k
        
        if Y is None:
            Y = X[1:]
            U = U[:-1]
            X = X[:-1]
        
        # Setup
        X = X.requires_grad_(True)
        OX = X
        
        if k is not None:
            # Center X and Y
            self.x_mean = torch.mean(OX, dim=0)
            OX_ = OX - self.x_mean
            Y_ = Y - self.x_mean
            
            # PCA on state data
            _, _, V = np.linalg.svd(OX_.detach().numpy())
            self.V = torch.tensor(V).requires_grad_(True)
            
            X = OX_ @ self.V[:,:k]
            Y = Y_ @ self.V[:,:k]
        
        if self.option == 1:
            #===========================#
            # OPTION 1: Continuous-time #
            #===========================#
            # Physics informed fitting
            self.x_dict_obj.fit(X)
            Z = self.x_dict_obj.transform(X)
            Z_ = Z.detach().cpu().numpy()
            
            F = self.pde_obj(OX)[:,:,None].detach().cpu().numpy()
            J, _ = jacobian_f(Z, OX)
            J = J.detach().cpu().numpy()
            JF = np.squeeze(J @ F, axis=2)
            
            A = np.linalg.lstsq(Z_, JF, rcond=None)[0].T
            exp_A = expm(self.dt * A)
            self.L1 = exp_A
        else:
            #=========================#
            # OPTION 2: Discrete-time #
            #=========================#
            # Physics informed fitting
            self.x_dict_obj.fit(X)
            Z = self.x_dict_obj.transform(X)
            Z_ = Z.detach().cpu().numpy()
            
            # Create fake time-advanced matrix from F
            # Use RK4 method
            # f1 = self.pde_obj(OX)
            # f2 = self.pde_obj(OX + self.dt/2 * f1)
            # f3 = self.pde_obj(OX + self.dt/2 * f2)
            # f4 = self.pde_obj(OX + self.dt * f3)
            # OX_prime = OX + self.dt * (f1 + 2*f2 + 2*f3 + f4) / 6
            OX_prime = OX + self.dt * self.pde_obj(OX)
            
            if k is not None:
                OX_prime_ = OX_prime - self.x_mean
                X_prime = OX_prime_ @ self.V[:,:k]
            else:
                X_prime = OX_prime
            
            Z_prime = self.x_dict_obj.transform(X_prime)
            Z_prime_ = Z_prime.detach().cpu().numpy()
            
            exp_A = np.linalg.lstsq(Z_, Z_prime_, rcond=None)[0].T
            self.L1 = exp_A
        
        #========================#
        # STEP 2: Control inputs #
        #========================#
        # Linear model fitting
        if self.xu_dict_obj is not None:
            self.xu_dict_obj.fit(torch.cat([X, U], dim=1))
            ZU = self.xu_dict_obj.transform(torch.cat([X, U], dim=1)).detach().cpu().numpy()
        else:
            ZU = U.detach().cpu().numpy()
        ZP = self.x_dict_obj.transform(Y).detach().cpu().numpy()
        
        B = np.linalg.lstsq(ZU, ZP - Z_ @ exp_A.T, rcond=None)[0].T
        self.L2 = B
        
        return ZP - Z_ @ exp_A.T
    
    def predict(self, x, U):
        '''
        Predict trajectory given a control input sequence and an initial state
        
        Args:
            x (torch.tensor): Initial state
            U (torch.tensor): Control input sequence in shape [N', D_INPUT]
        '''
        x_res = np.zeros((U.shape[0]+1, x.shape[0]))
        x_res[0,:] = x.detach().cpu().numpy()
            
        if self.k is not None:
            x_mean = self.x_mean.detach().numpy()
            V = self.V.detach().numpy()
            
            x = torch.flatten((x - self.x_mean)[None,:] @ self.V[:,:self.k])
        
        z = self.x_dict_obj.transform(x[None,:]).detach().cpu().numpy().T
        if self.xu_dict_obj is not None:
            zu = self.xu_dict_obj.transform(torch.cat([x[None,:], U[0,:][None,:]], dim=1))
        else:
            zu = U[0,:][None,:]
        
        L2_contribs = []
        for i in range(U.shape[0]):
            L2_contrib = self.L2 @ zu.detach().cpu().numpy().T
            L2_contribs.append(L2_contrib)
            
            z = self.L1 @ z + L2_contrib
            x_pred = self.x_dict_obj.untransform(torch.tensor(z.T)).detach().cpu().numpy()
            
            if self.k is not None:
                x_res[i+1,:] = x_pred @ V[:,:self.k].T + x_mean
            else:
                x_res[i+1,:] = x_pred
            
            if self.xu_dict_obj is not None:
                zu = self.xu_dict_obj.transform(torch.cat([torch.tensor(x_pred), U[i][None,:]], dim=1))
            else:
                zu = U[i][None,:]
        
        return x_res, L2_contribs
