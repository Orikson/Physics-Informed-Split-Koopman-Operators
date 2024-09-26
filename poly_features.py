''' 
File: `poly_features.py`
Description:
    Polynomial feature generation for EDMDc models
'''

from itertools import combinations_with_replacement
import numpy as np
import torch

#===========================#
# Numpy Polynomial Features #
#===========================#
class PolyFeatures_np:
    '''
    Polynomial feature generation for EDMDc models
    '''

    def __init__(self, order, mono=False):
        '''
        Initialize polynomial feature generator

        Args:
            order (int): Polynomial order
            mono (bool): Monomial features only
        '''
        self.order = order
        self.mono = mono
        
    def fit(self, X):
        '''
        Fit polynomial features

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
        if self.mono:
            return self._monomial_features(X)
        else:
            return self._polynomial_features(X)
    
    def untransform(self, Z):
        '''
        Untransform input data

        Args:
            Z (np.ndarray): Transformed data
        '''
        return Z[:,1:self.n_features+1]
    
    def _monomial_features(self, X):
        '''
        Generate monomial features

        Args:
            X (np.ndarray): Input data
        '''
        return np.concatenate([np.ones((X.shape[0], 1))] + [X**i for i in range(1, self.order+1)], axis=1)
    
    def _polynomial_features(self, X):
        '''
        Generate polynomial features

        Args:
            X (np.ndarray): Input data
        '''
        return np.concatenate([np.ones((X.shape[0], 1))] + [np.prod(np.array(list(combinations_with_replacement(X.T, i))), axis=1).T for i in range(1, self.order+1)], axis=1)

#===========================#
# Torch Polynomial Features #
#===========================#
class PolyFeatures_torch:
    '''
    Polynomial feature generation for EDMDc models
    '''

    def __init__(self, order, mono=False):
        '''
        Initialize polynomial feature generator

        Args:
            order (int): Polynomial order
            mono (bool): Monomial features only
        '''
        self.order = order
        self.mono = mono
        
    def fit(self, X):
        '''
        Fit polynomial features

        Args:
            X (torch.Tensor): Input data
        '''
        self.n_features = X.shape[1]
        
    def transform(self, X):
        '''
        Transform input data

        Args:
            X (torch.Tensor): Input data
        '''
        if self.mono:
            return self._monomial_features(X)
        else:
            return self._polynomial_features(X)
    
    def untransform(self, Z):
        '''
        Untransform input data

        Args:
            Z (torch.Tensor): Transformed data
        '''
        return Z[:,1:self.n_features+1]
    
    def _monomial_features(self, X):
        '''
        Generate monomial features

        Args:
            X (torch.Tensor): Input data
        '''
        return torch.cat([torch.ones(X.shape[0], 1)] + [X**i for i in range(1, self.order+1)], dim=1)

    def _polynomial_features(self, X):
        '''
        Generate polynomial features

        Args:
            X (torch.Tensor): Input data
        '''
        
        #return torch.cat([torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)] + [torch.prod(torch.combinations(X.T, r=i, with_replacement=True), dim=1).T for i in range(1, self.order+1)], dim=1)
        return torch.cat([torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)] + [torch.prod(torch.stack([torch.stack(v) for v in list(combinations_with_replacement(X.T, i))]), dim=1).T for i in range(1, self.order+1)], dim=1)
