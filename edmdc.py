''' 
File: `edmdc.py`
Description:
    Bilinear and Linear EDMDc
'''

import numpy as np

class EDMDc:
    '''
    Extended dynamic mode decomposition with control
    '''
    def __init__(self, dict_obj):
        '''
        Initialize EDMDc model
        
        Args:
            dict_obj (obj): Dictionary object for feature generation
        '''
        self.dict_obj = dict_obj
    
    def fit(self, X, U, Y=None, linear=False):
        '''
        Fit EDMDc model
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            Y (np.ndarray): State data matrix, in shape [N, D_STATE]. If None, set Y to timeshifted X
            linear (bool): If true, then learn a linear EDMDc model, otherwise, learn a bilinear model
        '''
        
        if Y is None:
            Y = X[1:]
            U = U[:-1]
            X = X[:-1]
            
        self.dict_obj.fit(X)

        if linear:
            self._fit_linear(X, U, Y)
        else:
            self._fit_bilinear(X, U, Y)
        self.linear = linear

    def predict(self, x, U):
        '''
        Predict trajectory given a control input sequence and an initial state
        
        Args:
            x (np.ndarray): Initial state
            U (np.ndarray): Control input sequence in shape [N', D_INPUT]
        '''
        x_res = np.zeros((U.shape[0]+1, x.shape[0]))
        x_res[0,:] = x
        
        if self.linear:
            _, zu = self._gen_linear(x[None,:], U[0,:][None,:])
        else:
            _, zu = self._gen_bilinear(x[None,:], U[0,:][None,:])
        
        for i in range(U.shape[0]):
            z = self.F @ zu.T
            x_res[i+1,:] = self.dict_obj.untransform(z.T)
            
            if self.linear:
                #_, zu = self._gen_linear(x_res[i+1][None,:], U[i][None,:])
                zu = self._gen_linear_Z(z.T, U[i][None,:])
            else:
                #_, zu = self._gen_bilinear(x_res[i+1][None,:], U[i][None,:])
                zu = self._gen_bilinear_Z(z.T, U[i][None,:])
        
        return x_res
        
    def _gen_linear(self, X, U):
        '''
        Generate full matrix for linear EDMDc model
        '''
        Z = self.dict_obj.transform(X)
        
        full_matrix = np.hstack([Z, U])
        return Z, full_matrix
    
    def _gen_linear_Z(self, Z, U):
        '''
        Generate full matrix for linear EDMDc model from Z
        '''
        full_matrix = np.hstack([Z, U])
        return full_matrix
    
    def _fit_linear(self, X, U, Y):
        '''
        Fit linear EDMDc model
        
        z_{k+1} = A z_k + B u_k
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            Y (np.ndarray): State data matrix, in shape [N, D_STATE]
        '''
        Z, full_matrix = self._gen_linear(X, U)
        ZP = self.dict_obj.transform(Y)
        
        self.D_Z = Z.shape[1]
        
        F = np.linalg.lstsq(full_matrix, ZP, rcond=None)[0].T
        
        self.F = F
        self.A, self.B = F[:,:self.D_Z], F[:,self.D_Z:]
    
    def _gen_bilinear(self, X, U):
        '''
        Generate full matrix for bilinear EDMDc model
        '''
        Z = self.dict_obj.transform(X)
        
        stack = [Z]
        for i in range(U.shape[1]):
            stack.append(U[:,i][:,None] * Z)
        full_matrix = np.hstack(stack)
        
        return Z, full_matrix

    def _gen_bilinear_Z(self, Z, U):
        '''
        Generate full matrix for bilinear EDMDc model from Z
        '''
        stack = [Z]
        for i in range(U.shape[1]):
            stack.append(U[:,i][:,None] * Z)
        full_matrix = np.hstack(stack)
        
        return full_matrix
    
    def _fit_bilinear(self, X, U, Y):
        '''
        Fit bilinear EDMDc model
        
        z_{k+1} = A z_k + sum_i u_{ki} B_i z_k
        
        Args:
            X (np.ndarray): State data matrix, in shape [N, D_STATE]
            U (np.ndarray): Input data matrix, in shape [N, D_INPUT]
            Y (np.ndarray): State data matrix, in shape [N, D_STATE]
        '''
        Z, full_matrix = self._gen_bilinear(X, U)
        ZP = self.dict_obj.transform(Y)
        
        self.D_Z = Z.shape[1]
        
        F = np.linalg.lstsq(full_matrix, ZP, rcond=None)[0].T
        
        self.F = F
        self.A, self.B = F[:,:self.D_Z], F[:,self.D_Z:]
        
        
