''' 
File: `fourier_features.py`
Description:
    Random fourier feature generation for EDMDc models
'''

import numpy as np
import torch

#========================#
# Numpy Fourier Features #
#========================#
class FourierFeatures_np:
    '''
    Random Fourier feature generation for EDMDc models
    '''

    def __init__(self, n_components, gamma, seed=None):
        '''
        Initialize Fourier feature generator

        Args:
            n_components (int): Number of Fourier features
            gamma (float): Width of Fourier features
            seed (int): Random seed
        '''
        self.n_components = n_components
        self.gamma = gamma
        self.seed = seed
        
    def fit(self, X):
        '''
        Fit Fourier features

        Args:
            X (np.ndarray): Input data
        '''
        self.n_features = X.shape[1]
        
        if self.seed is not None:
            np.random.seed(self.seed)
        
        self.W = np.sqrt(2 * self.gamma) * np.random.normal(size=(self.n_features, self.n_components))
        
    def transform(self, X):
        '''
        Transform input data

        Args:
            X (np.ndarray): Input data
        '''
        xw = X @ self.W
        return np.hstack([X, np.cos(xw), np.sin(xw)])
    
    def untransform(self, Z):
        '''
        Untransform input data

        Args:
            Z (np.ndarray): Transformed data
        '''
        return Z[:,:self.n_features]

#========================#
# Torch Fourier Features #
#========================#
class FourierFeatures_torch:
    '''
    Random Fourier feature generation for EDMDc models
    '''

    def __init__(self, n_components, gamma, seed=None):
        '''
        Initialize Fourier feature generator

        Args:
            n_components (int): Number of Fourier features
            gamma (float): Width of Fourier features
        '''
        self.n_components = n_components
        self.gamma = gamma
        self.seed = seed
        
    def fit(self, X):
        '''
        Fit Fourier features

        Args:
            X (torch.Tensor): Input data
        '''
        self.n_features = X.shape[1]
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # self.W = np.sqrt(2 * self.gamma) * torch.randn(self.n_features, self.n_components, dtype=X.dtype, device=X.device)
        self.W = np.sqrt(2 * self.gamma) * np.random.normal(size=(self.n_features, self.n_components))
        self.W = torch.tensor(self.W, dtype=X.dtype, device=X.device)
        
    def transform(self, X):
        '''
        Transform input data

        Args:
            X (torch.Tensor): Input data
        '''
        xw = X @ self.W
        return torch.cat([X, torch.cos(xw), torch.sin(xw)], dim=1)
    
    def untransform(self, Z):
        '''
        Untransform input data

        Args:
            Z (torch.Tensor): Transformed data
        '''
        return Z[:,:self.n_features]
