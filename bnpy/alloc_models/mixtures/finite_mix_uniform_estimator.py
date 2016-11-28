import numpy as np

import mix_local_dense_posterior_estimate
from bnpy.utils_array import argsort_bigtosmall_stable

default_hyper_kwargs = dict()
default_init_kwargs = dict(
    K=10,
    init_alloc_proba_K='uniform')

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
        proba_K=np.ones(K, dtype=np.float64) / K)
    info_dict = dict()
    return param_dict, info_dict


def calc_loss_from_summaries(
        GP,
        HP,
        SS,
        SS_for_loss=None,
        after_global_step=False):
    ''' Calculate loss

    Returns
    -------
    loss_val : float
    '''
    # Slack term
    # Only needed when not immediately after a global step.
    #loss_alloc_slack = -1 * (
    #    np.inner(SS['n_K'], np.log(GP['proba_K'])))

    # Entropy term
    # Negated because the loss = -1 * usual variational elbo objective
    if SS_for_loss is None:
        loss_alloc_entropy = 0.0
    else:
        loss_alloc_entropy = -1 * np.sum(SS_for_loss['H_resp_K'])
    return loss_alloc_entropy


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
        E_log_alloc_proba_K=np.log(GP['proba_K']),
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
    return param_dict

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