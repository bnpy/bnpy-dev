import numpy as np

import gauss_initializer_hypers
from wishart_expectations import E_inv_var_D, E_var_D
from gauss_wishart_expectations import \
    c_GaussWish_iid, LOG_TWO_PI, \
    E_logdetL_D, E_L_D, E_Lm_D, E_Lm2_D, E_mahalanobis_distance_N

import gauss_initializer_fromscratch

default_hyper_kwargs = dict(
    nu=0.0,
    kappa=0.00001,
    beta_D=None,
    m_D=None,
    prior_mean_x=0.0,
    prior_covar_x='0.1*eye')
default_init_kwargs = dict(
    K=10,
    seed=42,
    init_procedure='LP_from_rand_examples',
    init_fromscratch_module='gauss_initializer_fromscratch',
    init_model_path=None,
    )
default_global_step_kwargs = dict()
default_local_step_kwargs = dict()

def init_hyper_params(
        dataset,
        nu=default_hyper_kwargs['nu'],
        kappa=default_hyper_kwargs['kappa'],
        beta_D=default_hyper_kwargs['beta_D'],
        m_D=default_hyper_kwargs['m_D'],
        prior_mean_x=default_hyper_kwargs['prior_mean_x'],
        prior_covar_x=default_hyper_kwargs['prior_covar_x'],
        **hyper_kwargs):
    ''' Initialize hyperparameters

    Returns
    -------
    hyper_dict : dict with arrays
        nu, beta_D, m_D, kappa

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> X_ND = prng.randn(10000, 3)
    >>> hyper_dict = init_hyper_params(
    ...     X_ND, prior_mean_x=0.0, prior_covar_x='1*covdata')
    >>> hyper_dict['nu']
    5.0

    # Expected variance parameter under Wishart prior
    >>> E_var_D(**hyper_dict)
    array([ 0.98712142,  0.99059401,  0.97862501])
    '''
    prior_mean_D, prior_covar_DD = \
        gauss_initializer_hypers.init_prior_mean_and_covar(
            dataset, prior_mean_x, prior_covar_x)
    D = prior_mean_D.size
    nu = np.maximum(nu, D + 2)
    kappa = np.maximum(kappa, 0)
    if beta_D is None:
        beta_D = np.diag(prior_covar_DD) * (nu - 2)
    else:
        beta_D = as1D(toCArray(beta_D))
    if m_D is None:
        m_D = prior_mean_D
    else:
        m_D = as1D(toCArray(prior_mean_D))
    return dict(nu=nu, beta_D=beta_D, m_D=m_D, kappa=kappa)

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
        * nu_K
        * kappa_K
        * m_KD
        * beta_KD
    info_dict : dict with information about initialization process

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> X_a_NaD = 0.1 * prng.randn(500, 3) - 5
    >>> X_b_NbD = 0.1 * prng.randn(500, 3) + 0
    >>> X_c_NcD = 0.1 * prng.randn(5, 3) + 5
    >>> X_ND = np.vstack([X_a_NaD, X_b_NbD, X_c_NcD])
    >>> hyper_dict = init_hyper_params(
    ...     X_ND, prior_mean_x=0.0, prior_covar_x='0.1*eye')
    >>> param_dict, _ = init_global_params(
    ...     X_ND, hyper_dict,
    ...     init_procedure='LP_from_rand_examples',
    ...     K=3, seed=0)
    >>> param_dict['m_KD']
    array([[-0.07846796, -0.03355773,  0.18961633],
           [-0.114189  , -0.13109573, -0.15329057],
           [-5.14214057, -4.80015639, -5.08560407]])

    # Try again, using distance-biased weighting
    >>> param_dict, _ = init_global_params(
    ...     X_ND, hyper_dict,
    ...     init_procedure='LP_from_rand_examples_by_dist',
    ...     K=3, seed=0)
    >>> param_dict['m_KD']
    array([[-0.08868955,  0.0878295 ,  0.00864517],
           [-5.14214057, -4.80015639, -5.08560407],
           [ 5.0003675 ,  4.85160233,  4.85197192]])
    '''
    if init_model_path is None:
        if init_procedure.startswith('LP'):
            LP, info_dict = gauss_initializer_fromscratch.init_local_params(
                dataset,
                init_procedure=init_procedure,
                **init_kwargs)
            SS = summarize_local_params_for_update(dataset, LP)
            param_dict = update_global_params_from_summaries(
                SS, None, hyper_dict)
        else:
            param_dict = gauss_initializer_fromscratch.init_global_params(
                dataset,
                init_procedure=init_procedure,
                **init_kwargs)
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
    >>> X_ND = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_ND, LP)
    >>> hyper_dict = init_hyper_params(X_ND)
    >>> param_dict = update_global_params_from_summaries(SS, None, hyper_dict)
    >>> loss_v1 = calc_loss_from_summaries(
    ...     param_dict, hyper_dict, SS, after_global_step=1)
    >>> loss_v2 = calc_loss_from_summaries(
    ...     param_dict, hyper_dict, SS, after_global_step=0)
    >>> assert loss_v1 == loss_v2

    # Now look at loss with slightly different value for "nu" param
    # Loss should be slightly higher (worse) than ideal params
    >>> bad_params = param_dict.copy()
    >>> bad_params['nu_K'] += 0.1
    >>> bad_loss = calc_loss_from_summaries(
    ...     bad_params, hyper_dict, SS, after_global_step=0)
    >>> bad_loss > loss_v1
    True

    # Now look at loss with slightly different value for "m" param
    # Loss should be slightly higher (worse) than ideal params
    >>> bad_params = param_dict.copy()
    >>> bad_params['m_KD'] += 0.001
    >>> bad_loss = calc_loss_from_summaries(
    ...     bad_params, hyper_dict, SS, after_global_step=0)
    >>> bad_loss > loss_v1
    True
    '''
    PD = param_dict
    HD = hyper_dict
    K = SS['n_K'].size
    D = SS['x_KD'].shape[1]
    loss_K = 0.5 * D * LOG_TWO_PI * SS['n_K']
    for k in xrange(K):
        loss_K[k] = (
            - c_GaussWish_iid(
                nu=PD['nu_K'][k],
                kappa=PD['kappa_K'][k],
                beta_D=PD['beta_KD'][k],
                m_D=PD['m_KD'][k])
            + c_GaussWish_iid(**hyper_dict))
        if not after_global_step:
            w_logdetL = SS['n_K'][k] + HD['nu'] - PD['nu_K'][k]
            w_L_D = SS['xx_KD'][k] + HD['beta_D'] \
                + HD['kappa'] * np.square(HD['m_D']) \
                - PD['beta_KD'][k] \
                - PD['kappa_K'][k] * np.square(PD['m_KD'][k])
            w_Lm_D = SS['x_KD'][k] + HD['kappa'] * HD['m_D'] \
                - PD['kappa_K'][k] * PD['m_KD'][k]
            w_Lm2 = SS['n_K'][k] + HD['kappa'] - PD['kappa_K'][k]
            loss_K[k] -= (
                + 0.5 * w_logdetL * np.sum(
                    E_logdetL_D(
                        nu=PD['nu_K'][k],
                        beta_D=PD['beta_KD'][k]))
                - 0.5 * np.inner(w_L_D, E_L_D(
                        nu=PD['nu_K'][k],
                        beta_D=PD['beta_KD'][k]))
                + np.inner(w_Lm_D, E_Lm_D(
                        nu=PD['nu_K'][k],
                        beta_D=PD['beta_KD'][k],
                        m_D=PD['m_KD'][k]))
                - 0.5 * w_Lm2 * np.sum(E_Lm2_D(
                        nu=PD['nu_K'][k],
                        beta_D=PD['beta_KD'][k],
                        m_D=PD['m_KD'][k],
                        kappa=PD['kappa_K'][k]))
                )
    if return_vec:
        return loss_K
    return np.sum(loss_K)

# GLOBAL STEP
def update_global_params_from_summaries(
        SS, param_dict, hyper_dict,
        fill_existing_arrays=False,
        **kwargs):
    ''' Compute optimal global parameters from summaries

    Returns
    -------
    param_dict : dict of arrays which define variational posterior

    Examples
    --------
    >>> X_ND = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_ND, LP)
    >>> hyper_dict = init_hyper_params(X_ND)
    >>> param_dict = update_global_params_from_summaries(SS, None, hyper_dict)
    >>> param_dict['m_KD']
    array([[ 1.99998,  0.99999,  0.99999],
           [ 0.99999,  1.99998,  0.99999],
           [ 0.99999,  0.99999,  1.99998]])
    >>> param_dict['beta_KD']
    array([[ 0.30004,  0.30001,  0.30001],
           [ 0.30001,  0.30004,  0.30001],
           [ 0.30001,  0.30001,  0.30004]])
    '''
    if not fill_existing_arrays:
        nu_K = SS['n_K'] + hyper_dict['nu']
        kappa_K = SS['n_K'] + hyper_dict['kappa']
        m_KD = (hyper_dict['kappa'] * hyper_dict['m_D'] + SS['x_KD']) \
            / kappa_K[:,np.newaxis]
        beta_KD = hyper_dict['beta_D'] + SS['xx_KD'] \
            + hyper_dict['kappa'] * np.square(hyper_dict['m_D']) \
            - kappa_K[:,np.newaxis] * np.square(m_KD)
        return dict(
            nu_K=nu_K, kappa_K=kappa_K, m_KD=m_KD, beta_KD=beta_KD)
    else:
        GP = param_dict
        K = int(SS['n_K'].size)
        GP['nu_K'][:K] = SS['n_K'] + hyper_dict['nu']
        GP['kappa_K'][:K] = SS['n_K'] + hyper_dict['kappa']
        kappa_K = GP['kappa_K'][:K]
        GP['m_KD'][:K] = (
            hyper_dict['kappa'] * hyper_dict['m_D'] + SS['x_KD']) \
                / kappa_K[:,np.newaxis]
        m_KD = GP['m_KD'][:K] 
        GP['beta_KD'][:K] = hyper_dict['beta_D'] + SS['xx_KD'] \
            + hyper_dict['kappa'] * np.square(hyper_dict['m_D']) \
            - kappa_K[:,np.newaxis] * np.square(m_KD)
        return GP

# LOCAL STEP
def calc_log_proba(
        dataset_or_X, param_dict, **kwargs):
    ''' Compute log probability of observed data under each cluster

    Returns
    -------
    log_proba_NK : 2D array, size N x K

    Examples
    --------
    >>> X_ND = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_ND, LP)
    >>> hyper_dict = init_hyper_params(X_ND)
    >>> param_dict = update_global_params_from_summaries(SS, None, hyper_dict)
    >>> calc_log_proba(X_ND, param_dict)
    array([[ -0.02704412, -20.02517771, -20.02517771],
           [-20.02517771,  -0.02704412, -20.02517771],
           [-20.02517771, -20.02517771,  -0.02704412]])
    '''
    PD = param_dict
    if isinstance(dataset_or_X, np.ndarray):
        X_ND = dataset_or_X
    else:
        X_ND = dataset_or_X.X
    N, D = X_ND.shape
    K = PD['m_KD'].shape[0]

    log_proba_NK = np.zeros((N, K))
    for k in xrange(K):
        E_logdetL_k_D = E_logdetL_D(nu=PD['nu_K'][k], beta_D=PD['beta_KD'][k])
        log_proba_NK[:, k] = \
            - 0.5 * D * LOG_TWO_PI \
            + 0.5 * np.sum(E_logdetL_k_D) \
            - 0.5 * E_mahalanobis_distance_N(
                X_ND, 
                nu=PD['nu_K'][k], 
                beta_D=PD['beta_KD'][k],
                m_D=PD['m_KD'][k],
                kappa=PD['kappa_K'][k])
    return log_proba_NK

def summarize_local_params_for_update(dataset_or_X, LP, **kwargs):
    ''' Compute summary statistics from local params

    Examples
    --------
    >>> X_ND = np.eye(3) + np.ones((3, 3))
    >>> LP = dict(resp_NK=np.eye(3))
    >>> SS = summarize_local_params_for_update(X_ND, LP)
    >>> SS['n_K']
    array([ 1.,  1.,  1.])
    >>> SS['x_KD']
    array([[ 2.,  1.,  1.],
           [ 1.,  2.,  1.],
           [ 1.,  1.,  2.]])
    >>> SS['xx_KD']
    array([[ 4.,  1.,  1.],
           [ 1.,  4.,  1.],
           [ 1.,  1.,  4.]])
    '''
    if isinstance(dataset_or_X, np.ndarray):
        X_ND = dataset_or_X
    else:
        X_ND = dataset_or_X.X
    N, D = X_ND.shape

    if 'resp_NK' in LP:
        # Dense responsibility case
        resp_NK = LP['resp_NK']
        K = resp_NK.shape[1]
        # 1/2: Compute mean statistic
        x_KD = np.dot(resp_NK.T, X_ND)
        # 2/2: Compute expected outer-product statistic
        xx_KD = np.dot(resp_NK.T, np.square(X_ND))
        n_K = np.sum(resp_NK, axis=0)
    else:
        raise NotImplementedError("TODO")
    SS = dict(
        n_K=n_K,
        x_KD=x_KD,
        xx_KD=xx_KD)
    return SS