''' 
File: `d_ss_pi_edmdc.py`
Description:
    Delay coordinate physics informed extended dynamic mode decomposition with control.
    Uses a PDE prior to acquire a more accurate model.

    This form explicitly uses the Strang splitting form, although additional
        special considerations may be necessary when used in practice
        (such as those made in `dpi_edmdc.py`)
'''

import numpy as np
import scipy.linalg as linalg
from sklearn.linear_model import Lasso, Ridge

class D_SS_PI_EDMDc:
    def __init__(self, pde, dt, lmbda1, lmbda2, delay):
        '''
        Linear strang splitting PI-EDMDc with delay coordinates. 
        
        The way this works is we have a partially known pde x' = f(x,u) + g(x,u) where f(x,u) is known and g(x,u) is unknown.
        We learn the solution operator to f(x,u) in linear space
        Then, we learn the solution operator to g(x,u) in linear space
        
        Args:
            pde (function): the function f(x,u)
            dt (float): timestep separation in datasets
            lmbda1 (float): lambda value for K1 LASSO regression
            lmbda2 (float): lambda value for K2 LASSO regression
            delay (int): how many delay coordinates are used
        '''
        self.pde = pde
        self.dt = dt
        self.lmbda1 = lmbda1
        self.lmbda2 = lmbda2
        self.delay = delay
    
    def fit(self, X1, U1, Y1, X2, U2, ND=10, K1_OVERRIDE=None, **kwargs):
        '''
        Fit the model
        
        Args:
            X1 (np.ndarray): Part of dataset D1. State matrix in shape [N1, D_STATE*(delay+1)]
            U1 (np.ndarray): Part of dataset D1. Input matrix in shape [N1, D_INPUT]
            Y1 (np.ndarray): Part of dataset D1. Timeshifted state matrix in shape [N1, D_STATE]
            X2 (np.ndarray): Part of dataset D2. State matrix of collocation points in shape [N2, D_STATE*(delay+1)]
            U2 (np.ndarray): Part of dataset D2. Input matrix of collocation points in shape [N2, D_INPUT].
        '''
        lmbda1 = self.lmbda1
        lmbda2 = self.lmbda2
        D_INPUT = U1.shape[1]
        D_STATE = Y1.shape[1]

        self.N1 = X1.shape[0]
        self.N2 = X2.shape[0]

        self.D_INPUT = D_INPUT
        self.D_STATE = D_STATE
        
        #=====================#
        # Compute K1 using D2 #
        #=====================#
        if K1_OVERRIDE is None:
            # Compute fake timeshifted matrix using Euler update
            # dt = self.dt / 2
            dt = self.dt
            Y2 = X2[:,:D_STATE]
            for i in range(ND):
                Y2 = Y2 + dt / ND * self.pde(Y2)
            # Y2 = X2 + (self.dt / 2) * self.pde(X2)
            
            # Compute K1 using least squares
            # K1 = linalg.lstsq(X2, Y2)[0].T
            clf = Lasso(alpha=lmbda1, fit_intercept=False, **kwargs)
            # clf = Ridge(alpha=lmbda1, fit_intercept=False, **kwargs)
            clf.fit(X2, Y2)
            K1 = clf.coef_
            self.K1 = K1
        else:
            K1 = K1_OVERRIDE
            self.K1 = K1
        
        #=====================#
        # Compute K2 using D1 #
        #=====================#
        # Compute K1 inverse
        K1_inv = linalg.pinv(K1)
        
        # Compute K2 using least squares
        # K2 = linalg.lstsq(np.concatenate((X1 @ K1.T, X1, U1), axis=1), Y1 @ K1_inv.T)[0].T
        clf = Lasso(alpha=lmbda2, fit_intercept=False, **kwargs)
        # clf = Ridge(alpha=lmbda2, fit_intercept=False, **kwargs)
        clf.fit(np.concatenate((X1 @ K1.T, X1, U1), axis=1), Y1 @ K1_inv.T)
        K2 = clf.coef_
        self.K2 = K2
    
    def predict(self, x, U):
        '''
        Given a state and a list of control inputs, predict the trajectory along the given control inputs
        
        Args:
            x (np.ndarray): Initial state in shape [D_STATE*(delay+1)]
            U (np.ndarray): Control inputs in shape [N, D_INPUT]
        '''
        D_INPUT = self.D_INPUT
        D_STATE = self.D_STATE
        
        # Initialize state matrix
        X = np.zeros((U.shape[0]+1, D_STATE))
        X[0] = x[:D_STATE]
        
        # Iterate through control inputs
        for i in range(U.shape[0]):
            # Update using K1
            x1 = np.concatenate((self.K1 @ x[:,None], x[:,None], U[i][:,None]), axis=0)

            # Update using K2
            x2 = self.K1 @ self.K2 @ x1
            
            # Update state
            X[i+1] = x2.flatten()

            # Shift x
            tx = np.zeros_like(x)
            tx[D_STATE:] = x[:-D_STATE]
            tx[:D_STATE] = X[i+1]
            x = tx
        
        return X