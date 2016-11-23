import numpy as np
from scipy.special import digamma, gammaln

import topics_local_dense_posterior_estimate

from bnpy.utils_array import argsort_bigtosmall_stable
'''
from dp_mix_vb_proposal_planner import \
    make_plans_for_proposals, \
    make_summary_seeds_for_proposals, \
    evaluate_proposals, \
    default_split_kwargs, \
    default_merge_kwargs
'''

default_hyper_kwargs = dict(
    gamma=1.0,
    alpha=0.5)

default_init_kwargs = dict(
    K=10,
    init_alloc_proba_K='prior')

default_global_step_kwargs = dict()
default_local_step_kwargs = dict(
    local_step_mod='dense_posterior_estimate',
    )
local_step_module_map = dict(
    dense_posterior_estimate=topics_local_dense_posterior_estimate)

# INITIALIZE STATE
def init_hyper_params(data=None, gamma=1.0, alpha=0.5, **hyper_kwargs):
    return dict(gamma=gamma, alpha=alpha)

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
        rho_K=np.ones(K),
        omega_K=hyper_dict['gamma'] * np.ones(K))
    info_dict = dict()
    return param_dict, info_dict

def calc_loss_from_summaries(
        GP,
        HP,
        SS,
        SS_loss=None,
        after_global_step=False):
    ''' Calculate loss

    Returns
    -------
    loss_val : float

    Examples
    --------
    >>> HP = init_hyper_params()
    >>> GP, _ = init_global_params(None, HP, K=3)

    # Verify with non-trivial counts, loss drops after global step
    >>> SS = dict(n_K=np.asarray([10., 10., 10.]))
    >>> calc_loss_from_summaries(
    ...     GP, HP, SS, None)
    '''
    rho_K = GP['rho_K']
    omega_K = GP['omega_K']
    K = rho.size
    eta_1_K = rho_K * omega_K
    eta_0_K = (1 - rho_K) * omega_K
    digammaBoth = digamma(eta_1_K + eta_0_K)
    E_log_u_K = digamma(eta_1_K) - digammaBoth
    E_log_1mu_K = digamma(eta_0_K) - digammaBoth

    Ltop_c_p = K * c_Beta(1, gamma)
    Ltop_c_q = - c_Beta(eta1, eta0)
    Ltop_cDiff = Ltop_c_p + Ltop_c_q
    Ltop_logpDiff = np.inner(1.0 - eta_1_K, E_log_u_K) + \
        np.inner(gamma - eta_0_K, E_log_1mu_K)

    LcDsur_const = nDoc * K * np.log(alpha)
    LcDsur_rhoomega = nDoc * np.sum(ElogU) + \
        nDoc * np.inner(kvec(K), Elog1mU)
    L_alloc_rho = Ltop_cDiff + Ltop_logpDiff + LcDsur_const + LcDsur_rhoomega


    L_entropy = np.sum(SS_loss['H_resp_K'])

    # Cumulant term
    L_alloc_cDir_theta = -1 * (
        SS_loss['gammaln_sum_theta']
        - SS_loss['gammaln_theta_K']
        - SS_loss['gammaln_theta_rem'])

    # Slack term which depends only on cached theta terms
    L_alloc_slack_theta = np.sum(SS_loss['slack_theta_K']) \
        + SS_loss['slack_theta_rem']

    # Slack term which depends on rho
    E_proba_K = calc_E_proba_K(rho_K)
    E_proba_gt_K = 1 - np.cumsum(E_proba_gt_K)    
    L_alloc_slack_rho = alpha * (
        np.inner(E_proba_K, SS_update['sum_log_pi_K']) +
        np.inner(E_proba_gt_K, SS_update['sum_log_pi_rem_K']))

    return (L_alloc_rho + L_entropy + L_alloc_cDir_theta
        + L_alloc_slack_theta + L_alloc_slack_rho)

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