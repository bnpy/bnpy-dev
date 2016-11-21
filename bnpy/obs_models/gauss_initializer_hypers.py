import numpy as np

from bnpy.utils_array.shape_util import as1D, as2D, as3D, toCArray

def init_prior_mean_and_covar(
        dataset_or_X,
        prior_mean_x=0.0,
        prior_covar_x='1*eye'):
    ''' Initialize array hyperparameters for priors for Gaussian likelihoods

    Returns
    -------
    prior_mean_D : 1D array, size D
    prior_covar_DD : 2D array, size D x D
        symmetric and positive definite

    Examples
    --------
    >>> X_ND = np.ones((5, 3))
    >>> mean_D, covar_DD = init_prior_mean_and_covar(
    ...     X_ND, prior_mean_x=3.0, prior_covar_x='.2*eye')
    >>> mean_D
    array([ 3.,  3.,  3.])
    >>> covar_DD
    array([[ 0.2,  0. ,  0. ],
           [ 0. ,  0.2,  0. ],
           [ 0. ,  0. ,  0.2]])
    '''
    if isinstance(dataset_or_X, np.ndarray):
        X_ND = dataset_or_X
    else:
        X_ND = dataset_or_X.X
    D = X_ND.shape[1]
    prior_mean_D = as1D(np.asarray(prior_mean_x, dtype=np.float64))
    if prior_mean_D.size == 1:
        prior_mean_D = np.tile(prior_mean_D, (D))

    if isinstance(prior_covar_x, str):
        scalar, procedure_name = prior_covar_x.split('*')
        if procedure_name == 'eye':
            prior_covar_DD = float(scalar) * np.eye(D)
        elif procedure_name == 'covdata':
            prior_covar_DD = float(scalar) * np.cov(X_ND.T, bias=1)
        else:
            raise ValueError(
                "Unrecognized value prior_covar_x=%s" % prior_covar_x)
    else:
        prior_covar_DD = as2D(np.asarray(prior_covar_x))

    return prior_mean_D, prior_covar_DD