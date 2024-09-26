''' 
File: `d_edmdc.py`
Description:
    Bilinear and Linear EDMDc with delay coordinates
'''

import numpy as np

class D_EDMDc:
    '''
    Extended dynamic mode decomposition with control and delay coordinates
    '''
    def __init__(self, dict_obj):
        '''
        Initialize EDMDc model
        
        Args:
            dict_obj (obj): Dictionary object for feature generation
        '''
        self.dict_obj = dict_obj
    
    def fit(self, X, U, Y, linear=False):
        '''
        Fit EDMDc model
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE * DELAY]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            Y (np.ndarray): State data matrix, in shape [N, D_STATE]
            linear (bool): If true, then learn a linear EDMDc model, otherwise, learn a bilinear model
        '''
        _, D_STATE = Y.shape
        DELAY = X.shape[1] // D_STATE
        
        self.D_STATE = D_STATE
        self.DELAY = DELAY
        self.linear = linear
        
        self.dict_obj.fit(X[:,:D_STATE])
        
        if linear:
            Z = self._gen_linear(X, U, D_STATE, DELAY)
        else:
            Z = self._gen_bilinear(X, U, D_STATE, DELAY)
        
        ZP = self.dict_obj.transform(Y)
        
        self.F = np.linalg.lstsq(Z, ZP, rcond=None)[0].T
    
    def predict(self, x, U):
        '''
        Predict trajectory given a control input sequence and an initial state
        
        Args:
            x (np.ndarray): Initial state with delay in shape [D_STATE * DELAY]
            U (np.ndarray): Control input sequence in shape [N', D_INPUT]
        '''
        x_res = np.zeros((U.shape[0]+1, self.D_STATE))
        x_res[0,:] = x[:self.D_STATE]
        
        delay = x[self.D_STATE:]
        
        if self.linear:
            zu = self._gen_linear(x[None,:], U[0,:][None,:], self.D_STATE, self.DELAY)
        else:
            zu = self._gen_bilinear(x[None,:], U[0,:][None,:], self.D_STATE, self.DELAY)
        
        for i in range(U.shape[0]):
            z = self.F @ zu.T
            x_res[i+1,:] = self.dict_obj.untransform(z.T)
            
            # Add delay
            delay[self.D_STATE:] = delay[:-self.D_STATE]
            delay[:self.D_STATE] = x_res[i+1,:]
            
            x_full = np.concatenate([x_res[i+1,:], delay])
            
            if self.linear:
                zu = self._gen_linear(x_full[None,:], U[i][None,:], self.D_STATE, self.DELAY)
            else:
                zu = self._gen_bilinear(x_full[None,:], U[i][None,:], self.D_STATE, self.DELAY)
            
        return x_res
    
    def _gen_linear(self, X, U, D_STATE, DELAY):
        '''
        Generate linear EDMDc matrices
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE * DELAY]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            D_STATE (int): State dimension
            DELAY (int): Delay dimension
        '''
        Z = []
        for i in range(DELAY):
            Z.append(self.dict_obj.transform(X[:,i*D_STATE:(i+1)*D_STATE]))
        Z.append(U)
        Z = np.concatenate(Z, axis=1)
        return Z
    
    def _gen_bilinear(self, X, U, D_STATE, DELAY):
        '''
        Generate bilinear EDMDc matrices
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE * DELAY]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            D_STATE (int): State dimension
            DELAY (int): Delay dimension
        '''
        Z = []
        for i in range(DELAY):
            Z.append(self.dict_obj.transform(X[:,i*D_STATE:(i+1)*D_STATE]))
        Z = np.concatenate(Z, axis=1)
        
        stack = [Z]
        for i in range(U.shape[1]):
            stack.append(U[:,i][:,None] * Z)
        Z = np.concatenate(stack, axis=1)
        
        return Z
        
        
        
        
        
        
