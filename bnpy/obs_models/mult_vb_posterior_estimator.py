import numpy as np
import copy
import mult_initializer_fromscratch

from mult_dir_expectations import calc_E_log_phi_VK, c_Dir
from bnpy.utils_array.shape_util import as1D, as2D, toCArray

default_hyper_kwargs = dict(
    lam_V=None,
    prior_scale=0.1,
    prior_mean_proba_V='uniform',
    )
default_init_kwargs = dict(
    K=10,
    seed=42,
    init_procedure='LP_from_rand_examples',
    init_fromscratch_module='mult_initializer_fromscratch',
    init_model_path=None,
    )
default_global_step_kwargs = dict()
default_local_step_kwargs = dict()

def init_hyper_params(
        dataset,
        lam_V=default_hyper_kwargs['lam_V'],
        **hyper_kwargs):
    ''' Initialize hyperparameters

    Returns
    -------
    hyper_dict : dict with arrays
        * lam_V : 1D array, size V

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> X_DV = prng.randn(5, 4)
    >>> hyper_dict = init_hyper_params(
    ...     X_DV, lam_V=0.1)
    >>> hyper_dict['lam_V']
    array([ 0.1,  0.1,  0.1,  0.1])
    '''
    if isinstance(dataset, np.ndarray):
        V = dataset.shape[1]
    else:
        V = dataset.n_vocabs
    lam_V = as1D(np.asarray(lam_V))
    if lam_V.size == 1:
        lam_V = np.tile(lam_V, V)
    assert lam_V.size == V
    return dict(lam_V=lam_V)

def init_global_params(
        dataset,
        hyper_dict,
        init_model_path=default_init_kwargs['init_model_path'],
        init_procedure=default_init_kwargs['init_procedure'],
        **init_kwargs):
    ''' Initialize global parameters defining variational posterior

    Returns
    -------
    param_dict : dict with arrays
        * lam_KV
    info_dict : dict with information about initialization process

    Examples
    --------
    # TODO
    '''
    if init_model_path is None:
        if init_procedure.startswith('LP'):
            LP, info_dict = mult_initializer_fromscratch.init_local_params(
                dataset,
                init_procedure=init_procedure,
                **init_kwargs)
            SS = summarize_local_params_for_update(dataset, LP)
            param_dict = update_global_params_from_summaries(
                SS, None, hyper_dict)
        else:
            param_dict = mult_initializer_fromscratch.init_global_params(
                dataset,
                init_procedure=init_procedure,
                **init_kwargs)
            # TODO convert from topics_KV to lam_KV???
    return param_dict, info_dict

def calc_loss_from_summaries(
        param_dict,
        hyper_dict,
        SS,
        SS_for_loss=None,
        after_global_step=False,
        return_vec=False,
        **kwargs):
    ''' Calculate loss value given current summaries and global params

    Returns
    -------
    loss_dict : dict of loss values

    Examples
    --------
    >>> X_NV = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_NV, LP)
    >>> hyper_dict = init_hyper_params(X_NV, lam_V=0.1)
    >>> param_dict = update_global_params_from_summaries(SS, None, hyper_dict)
    >>> loss_v1 = calc_loss_from_summaries(
    ...     param_dict, hyper_dict, SS, after_global_step=1)
    >>> loss_v2 = calc_loss_from_summaries(
    ...     param_dict, hyper_dict, SS, after_global_step=0)
    >>> assert loss_v1 == loss_v2
    >>> print loss_v1
    23.6930044522

    # Now look at loss with slightly different values for global params
    # Loss should be slightly higher (worse) than ideal params
    >>> bad_params = copy.deepcopy(param_dict)
    >>> bad_params['lam_KV'] += 0.01
    >>> bad_loss = calc_loss_from_summaries(
    ...     bad_params, hyper_dict, SS, after_global_step=0)
    >>> bad_loss > loss_v1
    True

    >>> bad_params = copy.deepcopy(param_dict)
    >>> bad_params['lam_KV'] -= 0.01
    >>> bad_loss = calc_loss_from_summaries(
    ...     bad_params, hyper_dict, SS, after_global_step=0)
    >>> bad_loss > loss_v1
    True
    '''
    PD = param_dict
    HD = hyper_dict
    K = SS['count_KV'].shape[0]
    loss_K = np.zeros(K)
    if not after_global_step:
        E_log_phi_VK = calc_E_log_phi_VK(lam_KV=PD['lam_KV'])
    for k in xrange(K):
        loss_K[k] = c_Dir(lam_V=PD['lam_KV'][k]) - c_Dir(lam_V=HD['lam_V'])
        if not after_global_step:
            loss_K[k] -= np.inner(
                SS['count_KV'][k] + HD['lam_V'] - PD['lam_KV'][k], 
                E_log_phi_VK[:, k])
    if return_vec:
        return loss_K
    return np.sum(loss_K)

# GLOBAL STEP
def update_global_params_from_summaries(SS, param_dict, hyper_dict):
    ''' Compute optimal global parameters from summaries

    Returns
    -------
    param_dict : dict of arrays which define variational posterior

    Examples
    --------
    >>> X_NV = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_NV, LP)
    >>> hyper_dict = init_hyper_params(X_NV, lam_V=0.1)
    >>> param_dict = update_global_params_from_summaries(SS, None, hyper_dict)
    >>> param_dict['lam_KV']
    array([[ 2.1,  1.1,  1.1],
           [ 1.1,  2.1,  1.1],
           [ 1.1,  1.1,  2.1]])
    '''
    lam_KV = SS['count_KV'] + hyper_dict['lam_V'][np.newaxis, :]
    return dict(lam_KV=lam_KV)

# LOCAL STEP
def calc_log_proba(
        dataset_or_X, param_dict, **kwargs):
    ''' Compute log probability of observed data under each cluster

    Returns
    -------
    log_proba_NK : 2D array, size N x K

    Examples
    --------
    >>> X_NV = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_NV, LP)
    >>> hyper_dict = init_hyper_params(X_NV, lam_V=0.1)
    >>> param_dict = update_global_params_from_summaries(SS, None, hyper_dict)
    >>> calc_log_proba(X_NV, param_dict)
    array([[-5.22824911, -6.13734002, -6.13734002],
           [-6.13734002, -5.22824911, -6.13734002],
           [-6.13734002, -6.13734002, -5.22824911]])
    '''
    PD = param_dict
    E_log_phi_VK = calc_E_log_phi_VK(**PD)
    if isinstance(dataset_or_X, np.ndarray):
        X_NV = dataset_or_X
        log_proba_NK = np.dot(X_NV, E_log_phi_VK)
    else:
        raise NotImplementedError("TODO")
    return log_proba_NK

def summarize_local_params_for_update(dataset_or_X, LP, **kwargs):
    ''' Compute summary statistics from local params

    Examples
    --------
    >>> X_NV = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_NV, LP)
    >>> SS['count_KV']
    array([[ 2.,  1.,  1.],
           [ 1.,  2.,  1.],
           [ 1.,  1.,  2.]])
    '''
    if isinstance(dataset_or_X, np.ndarray):
        X_NV = dataset_or_X
    else:
        X_NV = dataset_or_X.X
    N, V = X_NV.shape

    if 'resp_NK' in LP:
        # Dense responsibility case
        resp_NK = LP['resp_NK']
        K = resp_NK.shape[1]
        count_KV = np.dot(resp_NK.T, X_NV)
    else:
        raise NotImplementedError("TODO")
    SS = dict(
        count_KV=count_KV)
    return SS