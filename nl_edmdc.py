''' 
File: `nl_edmdc.py`
Description:
    Nonlinear EDMDc
'''

import numpy as np

class NL_EDMDc:
    '''
    Nonlinear extended dynamic mode decomposition with control
    '''
    def __init__(self, x_dict_obj, xu_dict_obj):
        '''
        Initialize EDMDc model
        
        Args:
            dict_obj (obj): Dictionary object for feature generation
        '''
        self.x_dict_obj = x_dict_obj
        self.xu_dict_obj = xu_dict_obj
    
    def fit(self, X, U, Y=None, linear=None):
        '''
        Fit EDMDc model
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            Y (np.ndarray): State data matrix, in shape [N, D_STATE]. If None, set Y to timeshifted X
            linear (bool): Dummy argument for compatability
        '''
        
        if Y is None:
            Y = X[1:]
            U = U[:-1]
            X = X[:-1]
        
        self.x_dict_obj.fit(X)
        if self.xu_dict_obj is not None:
            self.xu_dict_obj.fit(np.concatenate([X, U], axis=1))
        
        Z = self.x_dict_obj.transform(X)
        if self.xu_dict_obj is not None:
            ZU= self.xu_dict_obj.transform(np.concatenate([X, U], axis=1))
        else:
            ZU= U
        ZP= self.x_dict_obj.transform(Y)
        
        ZF = np.concatenate([Z, ZU], axis=1)
        
        self.F = np.linalg.lstsq(ZF, ZP, rcond=None)[0].T
        
    def predict(self, x, U):
        '''
        Predict next state
        
        Args:
            x (np.ndarray): Initial data state [D_STATE]
            U (np.ndarray): Input data sequence, in shape [N', D_INPUT]
        '''
        x_res = np.zeros((U.shape[0]+1, x.shape[0]))
        x_res[0,:] = x
        
        z = self.x_dict_obj.transform(x[None,:])
        if self.xu_dict_obj is not None:
            zu= self.xu_dict_obj.transform(np.concatenate([x, U[0,:]])[None,:])
        else:
            zu= U[0,:][None,:]
        
        for i in range(U.shape[0]):
            zp = self.F @ np.concatenate([z, zu], axis=1).T
            x_res[i+1,:] = self.x_dict_obj.untransform(zp.T)
            
            z = self.x_dict_obj.transform(x_res[i+1,:][None,:])
            if self.xu_dict_obj is not None:
                zu= self.xu_dict_obj.transform(np.concatenate([x_res[i+1,:], U[i,:]])[None,:])
            else:
                zu= U[i,:][None,:]
        
        return x_res
