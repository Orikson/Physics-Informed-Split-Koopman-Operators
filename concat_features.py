''' 
File: `concat_features.py`
Description:
    Concatenate features for EDMDc models
Author: Eron Ristich
Date: 7/15/24
'''

import numpy as np
import torch

#=======================#
# Numpy Concat Features #
#=======================#
class ConcatFeatures_np:
    '''
    Concatenate features for EDMDc models
    '''
    
    def __init__(self, features):
        '''
        Initialize feature concatenator
        
        Args:
            features (list): List of feature objects
        '''
        self.features = features
    
    def fit(self, X):
        '''
        Fit feature concatenator
        
        Args:
            X (np.ndarray): Input data
        '''
        for feature in self.features:
            feature.fit(X)
    
    def transform(self, X):
        '''
        Transform input data
        
        Args:
            X (np.ndarray): Input data
        '''
        transforms = [feature.transform(X) for feature in self.features]
        if not hasattr(self, 'n_features'):
            self.n_features = [transform.shape[1] for transform in transforms]
        return np.concatenate(transforms, axis=1)

    def untransform(self, Z):
        '''
        Untransform input data
        
        Args:
            Z (np.ndarray): Transformed data
        '''
        z = Z[:,:self.n_features[0]]
        
        return self.features[0].untransform(z)

#=======================#
# Torch Concat Features #
#=======================#

class ConcatFeatures_torch:
    '''
    Concatenate features for EDMDc models
    '''
    
    def __init__(self, features):
        '''
        Initialize feature concatenator
        
        Args:
            features (list): List of feature objects
        '''
        self.features = features
    
    def fit(self, X):
        '''
        Fit feature concatenator
        
        Args:
            X (torch.Tensor): Input data
        '''
        for feature in self.features:
            feature.fit(X)
    
    def transform(self, X):
        '''
        Transform input data
        
        Args:
            X (torch.Tensor): Input data
        '''
        transforms = [feature.transform(X) for feature in self.features]
        if not hasattr(self, 'n_features'):
            self.n_features = [transform.shape[1] for transform in transforms]
        return torch.cat(transforms, dim=1)

    def untransform(self, Z):
        '''
        Untransform input data
        
        Args:
            Z (torch.Tensor): Transformed data
        '''
        z = Z[:,:self.n_features[0]]
        
        return self.features[0].untransform(z)
