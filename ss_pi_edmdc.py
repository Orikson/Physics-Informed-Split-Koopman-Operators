''' 
File: `ss_pi_edmdc.py`
Description:
    Physics informed extended dynamic mode decomposition with control.
    Uses a PDE prior to acquire a more accurate model.

    This form explicitly uses the Strang splitting form, although additional
        special considerations may be necessary when used in practice
        (such as those made in `pi_edmdc.py`)
'''

import numpy as np
import scipy.linalg as linalg
from sklearn.linear_model import Lasso, Ridge

class SS_PI_EDMDc:
    def __init__(self, Theta_X, pde, dt, lmbda1, lmbda2):
        '''
        Linear strang splitting PI-EDMDc. 
        
        The way this works is we have a partially known pde x' = f(x,u) + g(x,u) where f(x,u) is known and g(x,u) is unknown.
        We learn the solution operator to f(x,u) in linear space
        Then, we learn the solution operator to g(x,u) in linear space
        
        Args:
            Theta_X (function): theta function
            pde (function): the function f(x,u)
            dt (float): timestep separation in datasets
            lmbda1 (float): lambda value for K1 LASSO regression
            lmbda2 (float): lambda value for K2 LASSO regression
        '''
        self.theta = Theta_X
        self.pde = pde
        self.dt = dt
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
    
    def fit(self, X1, U1, Y1, X2, U2, **kwargs):
        '''
        Fit the model
        
        Args:
            X1 (np.ndarray): Part of dataset D1. State matrix in shape [N1, D_STATE]
            U1 (np.ndarray): Part of dataset D1. Input matrix in shape [N1, D_INPUT]
            Y1 (np.ndarray): Part of dataset D1. Timeshifted state matrix in shape [N1, D_STATE]
            X2 (np.ndarray): Part of dataset D2. State matrix of collocation points in shape [N2, D_STATE]
            U2 (np.ndarray): Part of dataset D2. Input matrix of collocation points in shape [N2, D_INPUT].
        '''
        lmbda1 = self.lmbda1
        lmbda2 = self.lmbda2
        self.theta.fit(X1)
        D_INPUT = U1.shape[1]
        
        #=====================#
        # Compute K1 using D2 #
        #=====================#
        # Compute fake timeshifted matrix using Euler update
        dt = self.dt / 2
        Y2 = X2
        for i in range(10):
            Y2 += dt / 10 * self.pde(Y2)
        Y2 = X2 + (self.dt / 2) * self.pde(X2)
        
        # Compute theta matrices
        Theta_X2 = np.concatenate([self.theta.transform(X2)], axis=1)
        Theta_Y2 = np.concatenate([self.theta.transform(Y2)], axis=1)
        
        # Compute K1 using least squares
        # K1 = linalg.lstsq(Theta_X2, Theta_Y2)[0].T
        clf = Lasso(alpha=lmbda1, fit_intercept=False, **kwargs)
        # clf = Ridge(alpha=lmbda1, fit_intercept=False, **kwargs)
        clf.fit(Theta_X2, Theta_Y2)
        K1 = clf.coef_
        
        # Set last rows to zeros and identity matrix for U
        tmp = np.zeros((K1.shape[0]+D_INPUT,K1.shape[1]+D_INPUT))
        tmp[:-D_INPUT,:-D_INPUT] = K1
        tmp[-D_INPUT:,-D_INPUT:] = np.eye(D_INPUT)
        K1 = tmp
        self.K1 = K1
        
        #=====================#
        # Compute K2 using D1 #
        #=====================#
        # Compute theta matrices
        Theta_X1 = np.concatenate([self.theta.transform(X1), U1], axis=1)
        Theta_Y1 = np.concatenate([self.theta.transform(Y1), U1], axis=1)
        
        # Compute K1 inverse
        K1_inv = linalg.pinv(K1)
        
        # Compute K2 using least squares
        # K2 = linalg.lstsq(Theta_X1 @ K1.T, Theta_Y1 @ K1_inv.T)[0].T
        clf = Lasso(alpha=lmbda2, fit_intercept=False, **kwargs)
        # clf = Ridge(alpha=lmbda2, fit_intercept=False, **kwargs)
        clf.fit(Theta_X1 @ K1.T, Theta_Y1 @ K1_inv.T)
        K2 = clf.coef_
        # print('K2 alpha:', clf.alpha_)
        
        # Set last rows to zeros and identity matrix for U
        addendum = np.zeros((D_INPUT, K2.shape[1]))
        addendum[-D_INPUT:,-D_INPUT:] = np.eye(D_INPUT)
        K2[-D_INPUT:] = addendum
        self.K2 = K2
        
        #========================#
        # Compute and truncate K #
        #========================#
        # Compute K
        K = K1 @ K2 @ K1
        
        # Truncate K, assuming U does not evolve
        K = K[:-D_INPUT,:]
        self.K = K
    
    def predict(self, x, U):
        '''
        Given a state and a list of control inputs, predict the trajectory along the given control inputs
        
        Args:
            x (np.ndarray): Initial state in shape [D_STATE]
            U (np.ndarray): Control inputs in shape [N, D_INPUT]
        '''
        # Initialize state matrix
        X = np.zeros((U.shape[0]+1, x.shape[0]))
        X[0] = x
        
        # Iterate through control inputs
        for i in range(U.shape[0]):
            # Compute theta matrix
            theta = self.theta.transform(X[i][None,:])
            theta = np.concatenate([theta, U[i][None,:]], axis=1)
            
            # Update state
            X[i+1] = self.theta.untransform((self.K @ theta.T).T).flatten()
        
        return X
