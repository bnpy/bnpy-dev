import numpy as np
from scipy.special import digamma, gammaln

import mix_local_dense_posterior_estimate

from dp_mix_vb_proposal_planner import \
    make_plans_for_proposals, \
    make_summary_seeds_for_proposals, \
    evaluate_proposals, \
    default_split_kwargs, \
    default_merge_kwargs
from bnpy.utils_array import argsort_bigtosmall_stable

default_hyper_kwargs = dict(gamma=1.0)

default_init_kwargs = dict(
    K=10,
    init_alloc_proba_K='prior')

default_global_step_kwargs = dict()
default_local_step_kwargs = dict(
    local_step_mod='mix_local_dense_posterior_estimate',
    )
local_module_map = dict(
    mix_local_dense_posterior_estimate=mix_local_dense_posterior_estimate)

# INITIALIZE STATE
def init_hyper_params(data=None, gamma=1.0, **hyper_kwargs):
    return dict(gamma=gamma)

def init_global_params(
        data, hyper_dict,
        K=default_init_kwargs['K'],
        init_alloc_proba_K=default_init_kwargs['init_alloc_proba_K'],
        **init_kwargs):
    ''' Initialize global parameters the define posterior estimate

    Returns
    -------
    param_dict : dict
    info_dict : dict
    '''
    K = int(K)
    param_dict = dict(
        eta_1_K=np.ones(K),
        eta_0_K=hyper_dict['gamma'] * np.ones(K))
    info_dict = dict()
    return param_dict, info_dict

def to_common_params(GP, **kwargs):
    ''' Convert global params to common point estimate

    Returns
    -------
    proba_K : 1D array, size K
    '''
    return dict(proba_K=calc_E_alloc_proba_K(**GP))

def calc_loss_from_local_params(data, LP, param_dict, hyper_dict):
    pass

def calc_loss_from_summaries(
        param_dict,
        hyper_dict,
        SS,
        SS_for_loss=None,
        after_global_step=False):
    ''' Calculate loss

    Returns
    -------
    loss_val : float

    Examples
    --------
    >>> HD = init_hyper_params()
    >>> zero_PD, _ = init_global_params(None, HD, K=3)
    >>> assert np.allclose(zero_PD['eta_0_K'], HD['gamma'] * np.ones(3))

    # Verify with zero counts, loss is exactly zero
    >>> zero_SS = dict(n_K=np.asarray([0., 0., 0.]))
    >>> calc_loss_from_summaries(
    ...     zero_PD, HD, zero_SS, None, after_global_step=True)
    0.0
    >>> calc_loss_from_summaries(
    ...     zero_PD, HD, zero_SS, None, after_global_step=False)
    0.0

    # Verify with non-trivial counts, loss drops after global step
    >>> SS = dict(n_K=np.asarray([10., 10., 10.]))
    >>> opt_PD = update_global_params_from_summaries(SS, None, HD)
    >>> calc_loss_from_summaries(
    ...     zero_PD, HD, SS, None)
    60.000000000000014
    >>> calc_loss_from_summaries(
    ...     opt_PD, HD, SS, None)
    38.22140354461056
    '''
    PD = param_dict
    HD = hyper_dict
    # Cumulant term
    K = SS['n_K'].size
    loss_alloc_cumulant = \
        - K * c_Beta(1.0, HD['gamma']) \
        + c_Beta(PD['eta_1_K'], PD['eta_0_K'])
    # Slack term
    # Only needed when not immediately after a global step.
    if after_global_step:
        loss_alloc_slack = 0.0
    else:
        ngt_K = calc_ngt_K(SS['n_K'])
        E_log_u_K, E_log_1mu_K = E_log_u_K_and_log_1mu_K(**param_dict)
        loss_alloc_slack = -1 * (
            np.inner(SS['n_K'] + 1.0 - PD['eta_1_K'], E_log_u_K) +
            np.inner(ngt_K + HD['gamma'] - PD['eta_0_K'], E_log_1mu_K))
    # Entropy term
    # Negated because of loss = - log proba
    if SS_for_loss is None:
        loss_alloc_entropy = 0.0
    else:
        loss_alloc_entropy = -1 * np.sum(SS_for_loss['H_resp_K'])
    return loss_alloc_cumulant + loss_alloc_slack + loss_alloc_entropy


# LOCAL STEP
def calc_local_params(
        data=None,
        param_dict=None,
        hyper_dict=None,
        LP=None,
        local_step_mod=default_local_step_kwargs['local_step_mod'],
        **local_kwargs):
    '''
    '''
    GP = param_dict
    if isinstance(local_step_mod, str):
        local_step_mod = local_module_map[local_step_mod]
    return local_step_mod.calc_local_params(
        data,
        LP,
        E_log_alloc_proba_K=E_log_alloc_proba_K(
            eta_1_K=GP['eta_1_K'],
            eta_0_K=GP['eta_0_K']),
        **local_kwargs)

def summarize_local_params_for_update(
        data, LP, 
        local_step_mod=default_local_step_kwargs['local_step_mod'],
        **local_kwargs):
    '''
    '''
    if isinstance(local_step_mod, str):
        local_step_mod = local_module_map[local_step_mod]
    return local_step_mod.summarize_local_params_for_update(
        data, LP, **local_kwargs)

def summarize_local_params_for_loss(
        data, LP,
        local_step_mod=default_local_step_kwargs['local_step_mod'],
        **local_kwargs):
    '''
    '''
    if isinstance(local_step_mod, str):
        local_step_mod = local_module_map[local_step_mod]
    return local_step_mod.summarize_local_params_for_loss(
        data, LP, **local_kwargs)

# GLOBAL STEP
def update_global_params_from_summaries(
        SS, param_dict, hyper_dict,
        fill_existing_arrays=False):
    if fill_existing_arrays:
        GP = param_dict
        K = SS['n_K'].size
        GP['eta_1_K'][:K] = SS['n_K'] + 1.0
        GP['eta_0_K'][:K] = calc_ngt_K(SS['n_K']) + hyper_dict['gamma']
        return GP
    else:
        eta_1_K = SS['n_K'] + 1.0
        eta_0_K = calc_ngt_K(SS['n_K']) + hyper_dict['gamma']
        return dict(eta_1_K=eta_1_K, eta_0_K=eta_0_K)

def reorder_summaries_and_update_params(
        SS_update, SS_loss, GP, HP,
        cur_uids_K=None,
        fill_existing_arrays=True):
    ''' Reorder the clusters from largest to smallest

    Returns
    -------
    SS_update : dict
    SS_loss : dict
    GP : dict
    new_uids_K : 1D array, size K

    Examples
    --------
    >>> HP = init_hyper_params()
    >>> SSU = dict(n_K=np.asarray([10., 20., 5.]))
    >>> SSL = dict(H_resp_K=np.asarray([11., 13., 17.]))
    >>> GP = update_global_params_from_summaries(SSU, None, HP)
    >>> cur_uids_K = np.asarray([11, 22, 33])
    >>> SSU, SSL, GP, cur_uids_K, _ = reorder_summaries_and_update_params(
    ...     SSU, SSL, GP, HP, cur_uids_K)
    >>> cur_uids_K
    array([22, 11, 33])
    >>> SSU['n_K']
    array([ 20.,  10.,   5.])
    '''
    K = SS_update['n_K'].size
    bigtosmall_order_K = argsort_bigtosmall_stable(SS_update['n_K'])
    cur_order_K = np.arange(K)

    if cur_uids_K is None:
        cur_uids_K = np.arange(K)
    else:
        cur_uids_K = np.asarray(cur_uids_K)
    if not fill_existing_arrays:
        GP = copy.deepcopy(GP)
        SS_update = copy.deepcopy(SS_update)
        SS_loss = copy.deepcopy(SS_loss)
        cur_uids_K = cur_uids_K.copy()

    needs_change = not np.allclose(cur_order_K, bigtosmall_order_K)
    if needs_change:
        cur_uids_K = cur_uids_K[bigtosmall_order_K]
        for key, cur_arr in SS_update.items():
            if cur_arr.shape[0] == K:
                cur_arr[:] = cur_arr[bigtosmall_order_K]
        for key, cur_arr in SS_loss.items():
            if cur_arr.shape[0] == K:
                cur_arr[:] = cur_arr[bigtosmall_order_K]

        # Update DP allocation variational parameters
        # eta_1_K and eta_0_K
        GP = update_global_params_from_summaries(
            SS_update, GP, HP, fill_existing_arrays=True)
        # Obs model global params just get shuffled
        for key, cur_arr in GP.items():
            if key == 'eta_1_K' or key == 'eta_0_K':
                continue
            if cur_arr.shape[0] == K:
                cur_arr[:] = cur_arr[bigtosmall_order_K]

    return SS_update, SS_loss, GP, cur_uids_K, needs_change



# UTIL EXPECTATIONS
def calc_E_alloc_proba_K(eta_0_K=None, eta_1_K=None, **kwargs):
    ''' Compute expected probability of each cluster E[\pi_k]

    Will not sum to one, since will be missing "remainder" term

    Returns
    -------
    proba_K : 1D array, size K
        sum will be positive and less than 1.0

    Examples
    --------
    >>> calc_E_alloc_proba_K(eta_1_K=[1, 1], eta_0_K=[1, 1])
    array([ 0.5 ,  0.25])
    '''
    eta_1_K = np.asarray(eta_1_K, dtype=np.float64)
    eta_0_K = np.asarray(eta_0_K, dtype=np.float64)
    proba_K = eta_1_K / (eta_1_K + eta_0_K)
    proba_K[1:] *= np.cumprod(1.0 - proba_K[:-1])
    return proba_K

def E_log_alloc_proba_K(eta_0_K=None, eta_1_K=None, **kwargs):
    ''' Compute log probability of each cluster E[ log \pi_k ]

    Returns
    -------
    E_log_alloc_proba_K : 1D array, size K

    Examples
    --------
    >>> E_log_alloc_proba_K(eta_1_K=[1., 1., 1.], eta_0_K=[1., 1., 1.])
    array([-1., -2., -3.])
    '''
    eta_1_K = np.asarray(eta_1_K, dtype=np.float64)
    eta_0_K = np.asarray(eta_0_K, dtype=np.float64)
    E_log_u_K, E_log_1mu_K = E_log_u_K_and_log_1mu_K(
        eta_1_K=eta_1_K, eta_0_K=eta_0_K)
    # Do some in-place calculations
    E_log_alloc_proba_K = E_log_u_K
    E_log_alloc_proba_K[1:] += np.cumsum(E_log_1mu_K[:-1])
    return E_log_alloc_proba_K

def E_log_u_K_and_log_1mu_K(eta_0_K=None, eta_1_K=None, **kwargs):
    digamma_both_K = digamma(eta_1_K + eta_0_K)
    E_log_u_K = digamma(eta_1_K) - digamma_both_K
    E_log_1mu_K = digamma(eta_0_K) - digamma_both_K
    return E_log_u_K, E_log_1mu_K


def c_Beta(eta_1_K, eta_0_K):
    ''' Evaluate cumulant function of Beta distribution

    Parameters
    -------
    eta_1_K : 1D array, size K
        represents ON pseudo-count parameter of the Beta
    eta_0_K : 1D array, size K
        represents OFF pseudo-count parameter of the Beta

    Returns
    -------
    c : float
        = \sum_k c_B(eta_1_K[k], eta_0_K[k])
    '''
    return np.sum(gammaln(eta_1_K + eta_0_K) \
        - gammaln(eta_1_K) - gammaln(eta_0_K))


def calc_ngt_K(n_K):
    """ Convert count vector to vector of "greater than" counts.

    Args
    -------
    n_K : 1D array, size K
        each entry k represents the count of items assigned to comp k.

    Returns
    -------
    ngt_K : 1D array, size K
        each entry k gives the total count of items at index above k
        ngt_K[k] = np.sum(n_K[k:])

    Example
    -------
    >>> calc_ngt_K([1., 3., 7., 2])
    array([ 12.,   9.,   2.,   0.])
    """
    n_K = np.asarray(n_K, dtype=np.float64)
    ngt_K = np.zeros_like(n_K)
    ngt_K[:-1] = np.cumsum(n_K[::-1])[::-1][1:]
    return ngt_K