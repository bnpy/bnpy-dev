import numpy as np

def E_var_D(nu=None, beta_D=None, **kwargs):
    '''
    '''
    return beta_D / (nu - 2)

def E_inv_var_D(nu=None, beta_D=None, **kwargs):
    '''
    '''
    return nu / beta_D
