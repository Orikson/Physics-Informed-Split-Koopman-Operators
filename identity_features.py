''' 
File: `identity_features.py`
Description:
    Identity feature f(x) = x
'''

import numpy as np
import torch

#=========================#
# Numpy Identity Features #
#=========================#
class IdentityFeatures_np:
    '''
    Identity feature f(x) = x
    '''

    def __init__(self):
        '''
        Initialize identity feature generator
        '''
        pass
        
    def fit(self, X):
        '''
        Fit identity features

        Args:
            X (np.ndarray): Input data
        '''
        self.n_features = X.shape[1]
        
    def transform(self, X):
        '''
        Transform input data

        Args:
            X (np.ndarray): Input data
        '''
        return X
    
    def untransform(self, Z):
        '''
        Untransform input data

        Args:
            Z (np.ndarray): Transformed data
        '''
        return Z
    
#=========================#
# Torch Identity Features #
#=========================#
class IdentityFeatures_torch:
    '''
    Identity feature f(x) = x
    '''

    def __init__(self):
        '''
        Initialize identity feature generator
        '''
        pass
        
    def fit(self, X):
        '''
        Fit identity features

        Args:
            X (torch.tensor): Input data
        '''
        self.n_features = X.shape[1]
        
    def transform(self, X):
        '''
        Transform input data

        Args:
            X (torch.tensor): Input data
        '''
        return X
    
    def untransform(self, Z):
        '''
        Untransform input data

        Args:
            Z (torch.tensor): Transformed data
        '''
        return Z
