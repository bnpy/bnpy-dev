import numpy as np
from scipy.special import gammaln, digamma

LOG_TWO = np.log(2.0)
LOG_TWO_PI = np.log(2.0 * np.pi)

def E_mahalanobis_distance_N(
        X_ND, nu=None, beta_D=None, m_D=None, kappa=None, **kwargs):
    ''' Expected value of (x - mu)^T L (x - mu)

    Returns
    -------
    dist_N : 1D array, size N
        Distance vector. Values should always be positive.
    '''
    D = int(m_D.size)
    tmp_ND = X_ND - m_D
    np.square(tmp_ND, out=tmp_ND)
    dist_N = np.dot(tmp_ND, nu / beta_D)
    dist_N += D / kappa
    return dist_N

def E_logdetL_D(nu=None, beta_D=None, **kwargs):
    ''' Expected value of log of each value along diagonal of L

    Returns
    -------
    E_logdetL_D : 1D array, size D
    '''
    return LOG_TWO - np.log(beta_D) + digamma(0.5 * nu)

def E_L_D(nu=None, beta_D=None, **kwargs):
    ''' Expected value of diagonal of L

    Returns
    -------
    E_L_D : 1D array, size D
    '''
    return nu / beta_D

def E_Lm_D(nu=None, beta_D=None, m_D=None, **kwargs):
    ''' Expected value of np.dot(L, m)

    Returns
    -------
    E_Lm_D : 1D array, size D
    '''
    return (nu / beta_D) * m_D

def E_Lm2_D(nu=None, beta_D=None, m_D=None, kappa=None, **kwargs):
    ''' Expected value of np.dot(L, m * m)

    Returns
    -------
    E_Lm2_D : 1D array, size D
    '''
    return 1.0 / kappa + (nu / beta_D) * (m_D * m_D)

def c_GaussWish_iid(nu=None, beta_D=None, kappa=None, m_D=None, **kwargs):
    ''' Compute cumulant function for Gaussian-Wishart distribution

    Returns
    -------
    cumulant_val : scalar

    Examples
    --------
    >>> c_GaussWish_iid(nu=5.0, beta_D=[5.0], m_D=[0.0], kappa=0.0001)
    3.5180647599802954
    '''
    beta_D = np.asarray(beta_D)
    m_D = np.asarray(m_D)
    D = beta_D.size
    c = 0.5 * D * LOG_TWO_PI \
        - 0.5 * D * np.log(kappa) \
        - 0.5 * nu * np.sum(np.log(beta_D / 2.0)) \
        + D * gammaln(nu / 2.0)
    return c