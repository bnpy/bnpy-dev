import numpy as np

def summarize_local_params_for_update(data, LP, **kwargs):
    ''' Compute summary stats used for global param updates from local params

    Returns
    -------
    SS : dict with array fields
        * n_K
    '''
    if 'resp_NK' in LP:
        resp_NK = LP['resp_NK']
        n_K = np.sum(resp_NK, axis=0)
    else:
        raise NotImplementedError("TODO sparse_resp_NK")
    return dict(n_K=n_K)

def summarize_local_params_for_loss(data, LP, **kwargs):
    ''' Compute summary stats used only for loss calculation from local params

    Returns
    -------
    SS : dict with array fields
        * H_resp_K
    '''
    if 'resp_NK' in LP:
        resp_NK = LP['resp_NK']
        tmp_NK = np.log(resp_NK)
        tmp_NK *= resp_NK
        H_resp_K = -1 * np.sum(tmp_NK, axis=0)
    else:
        raise NotImplementedError("TODO sparse_resp_NK")
    return dict(H_resp_K=H_resp_K)


def calc_local_params(data, LP, E_log_alloc_proba_K=None,
        local_step_copy_large_arrays=False,
        local_step_min_resp_val=1e-100,
        **kwargs):
    ''' Compute local parameters

    Returns
    -------
    LP : dict with array fields
        * resp_NK
    '''
    if local_step_copy_large_arrays:
        log_proba_NK = LP['log_proba_NK'].copy()
    else:
        log_proba_NK = LP['log_proba_NK']        

    # TODO: allow option to avoid inplace operations
    # this would make things autograd friendly
    log_proba_NK += E_log_alloc_proba_K[np.newaxis, :]
    resp_NK = log_proba_NK
    resp_NK -= np.max(resp_NK, axis=1)[:, np.newaxis]
    np.exp(resp_NK, out=resp_NK)
    resp_NK /= resp_NK.sum(axis=1)[:, np.newaxis]
    # Avoid later np.log(0.0) badness by enforcing small min value
    np.maximum(resp_NK, local_step_min_resp_val, out=resp_NK)
    return dict(resp_NK=resp_NK)