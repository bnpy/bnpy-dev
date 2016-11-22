import numpy as np
import itertools

from bnpy.utils_array import uid2k

def make_plans_for_split_proposals(
        data=None, LP=None, SSU=None, SSL=None, GP=None, HP=None,
        cur_uids_K=None,
        uid_generator=None,
        s_select_procedure='all_clusters',
        **plan_kwargs):
    '''
    '''
    K = GP['eta_1_K'].size
    if cur_uids_K is None:
        cur_uids_K = np.arange(K)
    if s_select_procedure == 'all_clusters':
        plan_dict_list = list()
        for uid in cur_uids_K:
            cur_plan_dict = dict(
                uid=uid,
                new_uids=[uid_generator.next()
                    for k in xrange(int(plan_kwargs['s_Knew']))],
                proposal_type='split')
            plan_dict_list.append(cur_plan_dict)
    else:
        raise ValueError(
            "Unrecognized s_select_procedure: " + s_select_procedure)
    return plan_dict_list

def calc_seed_summaries_for_split_proposals(
        plan_dict_list=None,
        **s_kwargs):
    ''' Compute seed summaries

    Returns
    -------
    seed_list : list of dicts
        Each dict has fields:
        * uid
        * new_uids
        * proposal_type = 'split'
        * seed_SS_update
        * seed_SS_loss
        
    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> K = 5
    >>> 
    '''
    assert plan_dict_list is not None
    for plan_dict in plan_dict_list:
        if 'proposal_type' in plan_dict:
            if plan_dict['proposal_type'] != 'split':
                continue
        cur_plan_dict = s_kwargs.copy()
        cur_plan_dict.update(plan_dict)
        updated_plan_dict = calc_seed_dict_for_split_proposal(
            **cur_plan_dict)
        if updated_plan_dict is None:
            plan_dict['status'] = False
        else:
            plan_dict['status'] = True
            plan_dict.update(updated_plan_dict)

    return plan_dict_list

def make_full_summaries_from_seed_for_split_proposal(
        SS_update, SS_loss,
        uid=None,
        cur_uids_K=None,
        new_uids=None,
        seed_SS_update=None,
        seed_SS_loss=None,
        keep_empty_cluster=False,
        **kwargs):
    ''' Create complete summaries for updates and loss under merge proposal

    Examples
    --------
    >>> SS_update = dict(n_K=np.asarray([100., 200., 300.]))
    >>> SS_loss = dict(H_resp_K=np.asarray([10., 20., 30.]))
    >>> seed_SS_update = dict(n_K=[60, 30, 10.])
    >>> seed_SS_loss = dict(H_resp_K=[11., 13., 17.])
    >>> prop_SSU, prop_SSL, prop_uids_K = \\
    ...     make_full_summaries_from_seed_for_split_proposal(
    ...         SS_update, SS_loss,
    ...         uid=0,
    ...         new_uids=[-1, -2, -3],
    ...         seed_SS_update=seed_SS_update,
    ...         seed_SS_loss=seed_SS_loss)
    >>> prop_uids_K
    array([ 1,  2, -1, -2, -3])
    >>> prop_SSU['n_K']
    array([ 200.,  300.,   60.,   30.,   10.])
    >>> prop_SSL['H_resp_K']
    array([ 20.,  30.,  11.,  13.,  17.])

    # Try with keep_empty_cluster=True
    >>> prop_SSU, prop_SSL, prop_uids_K = \\
    ...     make_full_summaries_from_seed_for_split_proposal(
    ...         SS_update, SS_loss,
    ...         uid=0,
    ...         new_uids=[-1, -2, -3],
    ...         seed_SS_update=seed_SS_update,
    ...         seed_SS_loss=seed_SS_loss,
    ...         keep_empty_cluster=True)
    >>> prop_SSU['n_K']
    array([   0.,  200.,  300.,   60.,   30.,   10.])
    '''
    K = SS_update['n_K'].size
    if cur_uids_K is None:
        cur_uids_K = np.arange(K)
    k = uid2k(cur_uids_K, uid)

    if keep_empty_cluster:
        prop_uids_K = cur_uids_K
    else:
        prop_uids_K = np.delete(cur_uids_K, k)
    prop_uids_K = np.hstack([prop_uids_K, new_uids])

    # Set update statistics
    prop_SS_update = dict()
    for key, cur_arr in SS_update.items():
        if cur_arr.shape[0] == K:
            if keep_empty_cluster:
                prop_arr = cur_arr.copy()
                prop_arr[k] = 0.0
            else:
                prop_arr = np.delete(cur_arr, k, axis=0)
            prop_arr = np.append(prop_arr, seed_SS_update[key], axis=0)
            prop_SS_update[key] = prop_arr
        else:
            prop_SS_update[key] = cur_arr

    # Set loss statistics
    prop_SS_loss = dict()
    for key, cur_arr in SS_loss.items():
        if cur_arr.shape[0] == K:
            if keep_empty_cluster:
                prop_arr = cur_arr.copy()
                prop_arr[k] = 0.0
            else:
                prop_arr = np.delete(cur_arr, k, axis=0)
            prop_arr = np.append(prop_arr, seed_SS_loss[key], axis=0)
            prop_SS_loss[key] = prop_arr
        else:
            prop_SS_loss[key] = cur_arr
    return prop_SS_update, prop_SS_loss, prop_uids_K

def calc_seed_dict_for_split_proposal(
        data, *args, **s_kwargs):
    s_kwargs['return_rLP_only'] = True
    rLP = calc_local_params_for_split_proposal(
        data, *args, **s_kwargs)
    if rLP is None:
        return None
    summarize_local_params_for_update = \
        s_kwargs['summarize_local_params_for_update']
    summarize_local_params_for_loss = \
        s_kwargs['summarize_local_params_for_loss']
    seed_SS_update = summarize_local_params_for_update(data, rLP)
    seed_SS_loss = summarize_local_params_for_loss(data, rLP)
    plan_dict = dict(
        proposal_type='split',
        uid=s_kwargs['uid'],
        new_uids=s_kwargs['new_uids'],
        seed_SS_update=seed_SS_update,
        seed_SS_loss=seed_SS_loss)
    return plan_dict

def calc_local_params_for_split_proposal(
        data,
        LP=None,
        GP=None,
        HP=None,
        uid=None,
        cur_uids_K=None,
        new_uids=None,
        init_global_params=None,
        calc_local_params=None,
        summarize_local_params_for_update=None,
        update_global_params_from_summaries=None,
        local_step_kwargs=None,
        s_n_iters=10,
        s_min_n_examples=50.0,
        return_rLP_only=False,
        **s_kwargs):
    ''' Construct proposed local parameters for given split proposal

    Returns
    -------
    prop_LP : dict

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    '''
    if local_step_kwargs is None:
        local_step_kwargs = dict()
    if cur_uids_K is None:
        K = LP['resp_NK'].shape[1]
        cur_uids_K = np.arange(K)
    uid = uid
    k = uid2k(cur_uids_K, uid)

    # Sample restricted set
    Knew = len(new_uids)
    s_kwargs['K'] = Knew

    r_ids = np.flatnonzero(LP['resp_NK'][:, k] > .01)
    if len(r_ids) < s_min_n_examples:
        return None

    r_dataset = data.make_subset(r_ids)
    rGP, info_dict = init_global_params(
        r_dataset, HP, **s_kwargs)

    for riter in range(s_n_iters):
        if riter > 0:    
            rSS = summarize_local_params_for_update(data, rLP)
            rGP = update_global_params_from_summaries(rSS, rGP, HP)
        rLP = calc_local_params(data, rGP,
            local_step_resp_sum_N=LP['resp_NK'][:, k],
            **local_step_kwargs)

    if return_rLP_only:
        return rLP
    prop_resp_NK = np.delete(LP['resp_NK'], k, axis=1)
    prop_resp_NK = np.hstack([prop_resp_NK, rLP['resp_NK']])
    prop_uids_K = np.hstack([np.delete(cur_uids_K, k), new_uids])
    return dict(resp_NK=prop_resp_NK), prop_uids_K