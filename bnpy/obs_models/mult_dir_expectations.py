import numpy as np

from scipy.special import digamma, gammaln

def calc_E_log_phi_VK(lam_KV=None, **kwargs):
    ''' Expected log probability of each word under each topic

    Returns
    -------
    E_log_phi_VK : 2D array, size V x K

    Examples
    --------
    >>> lam_KV = 5 * np.eye(3) + 0.1 * np.ones((3, 3))
    >>> calc_E_log_phi_VK(lam_KV)
    array([[ -0.0424014 , -11.99416587, -11.99416587],
           [-11.99416587,  -0.0424014 , -11.99416587],
           [-11.99416587, -11.99416587,  -0.0424014 ]])
    '''
    E_log_phi_VK = lam_KV.T.copy()
    digamma(E_log_phi_VK, out=E_log_phi_VK)
    digammaColSumVec = digamma(np.sum(lam_KV, axis=1))
    E_log_phi_VK -= digammaColSumVec[np.newaxis,:]
    return E_log_phi_VK


def c_Dir(lam_V=None, **kwargs):
    ''' Evaluate cumulant function of Dirichlet

    Returns
    --------
    c : scalar real value of cumulant function at provided args
    '''
    assert lam_V.ndim == 1
    return gammaln(np.sum(lam_V)) - np.sum(gammaln(lam_V))
