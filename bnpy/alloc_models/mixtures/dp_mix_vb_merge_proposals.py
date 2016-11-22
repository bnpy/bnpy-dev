import numpy as np
import itertools

from bnpy.utils_array import uidpair2kpair

def make_plans_for_merge_proposals(
        data=None, LP=None, SSU=None, SSL=None, GP=None, HP=None,
        cur_uids_K=None,
        m_select_procedure='all_pairs',
        **plan_kwargs):
    '''
    '''
    K = GP['eta_1_K'].size
    if cur_uids_K is None:
        cur_uids_K = np.arange(K)
    if m_select_procedure == 'all_pairs':
        plan_dict_list = list()
        for uid_pair in itertools.combinations(cur_uids_K, 2):
            cur_plan_dict = dict(
                uid_pair=uid_pair,
                proposal_type='merge')
            plan_dict_list.append(cur_plan_dict)
    elif m_select_procedure == 'pairs_that_minimize_loss':
        raise NotImplementedError("TODO")
    else:
        raise ValueError(
            "Unrecognized m_select_procedure: " + m_select_procedure)
    return plan_dict_list

def calc_seed_summaries_for_merge_proposals(
        data=None, LP=None, SSU=None, SSL=None, GP=None, HP=None, 
        cur_uids_K=None,
        plan_dict_list=None,
        m_uid_pair_list=None,
        **m_kwargs):
    ''' Compute seed summaries

    Returns
    -------
    seed_list : list of dicts
        Each dict has fields:
        * uid_pair
        * proposal_type = 'merge'
        * seed_SS_update
        * seed_SS_loss

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> K = 5
    >>> LP = dict(resp_NK=prng.dirichlet(np.ones(K), size=2))
    >>> seed_list = calc_seed_summaries_for_merge_proposals(
    ...     data=None, LP=LP, m_uid_pair_list=[(0,1), (0, 2), (1, 4)])

    # Verify pair (0, 1)
    >>> seed_SS_loss_01 = seed_list[0]['seed_SS_loss']
    >>> seed_SS_loss_01['H_resp_kA']
    0.68191018644294066
    >>> calc_prop_H_resp_kA(LP['resp_NK'], 0, 1)
    0.68191018644294066

    # Verify pair (1, 4)
    >>> seed_SS_loss_14 = seed_list[-1]['seed_SS_loss']
    >>> seed_SS_loss_14['H_resp_kA']
    0.6384899604300408
    >>> calc_prop_H_resp_kA(LP['resp_NK'], 1, 4)
    0.6384899604300408
    '''
    if cur_uids_K is None:
        cur_uids_K = np.arange(LP['resp_NK'].shape[1])

    if plan_dict_list is None:
        plan_dict_list = list()
        for uid_pair in m_uid_pair_list:
            plan_dict_list.append(
                dict(
                    proposal_type='merge',
                    uid_pair=uid_pair))

    for plan_dict in plan_dict_list:
        if plan_dict['proposal_type'] != 'merge':
            continue
        uidA, uidB = plan_dict['uid_pair']
        kA, kB = uidpair2kpair(cur_uids_K, uidA, uidB)
        propseed_SS_loss = dict(
            H_resp_kA=calc_prop_H_resp_kA(LP['resp_NK'], kA, kB))
        plan_dict['seed_SS_update'] = None
        plan_dict['seed_SS_loss'] = propseed_SS_loss
        plan_dict['status'] = True
    return plan_dict_list

def make_full_summaries_from_seed_for_merge_proposal(
        SS_update, SS_loss,
        uid_pair=None,
        seed_SS_update=None,
        seed_SS_loss=None,
        cur_uids_K=None,
        **kwargs):
    ''' Create complete summaries for updates and loss under merge proposal

    Examples
    --------
    >>> SS_update = dict(n_K=np.asarray([100., 200., 300.]))
    >>> SS_loss = dict(H_resp_K=np.asarray([10., 20., 30.]))
    >>> seed_SS_update = None
    >>> seed_SS_loss = dict(H_resp_kA=11)
    >>> prop_SSU, prop_SSL, prop_uids_K = \\
    ...     make_full_summaries_from_seed_for_merge_proposal(
    ...         SS_update, SS_loss,
    ...         uid_pair=(0, 1),
    ...         seed_SS_update=seed_SS_update,
    ...         seed_SS_loss=seed_SS_loss)
    >>> prop_SSU['n_K']
    array([ 300.,  300.])
    >>> prop_SSL['H_resp_K']
    array([ 11.,  30.])
    '''
    K = SS_update['n_K'].size
    if cur_uids_K is None:
        cur_uids_K = np.arange(K)
    uidA, uidB = uid_pair
    try:
        kA, kB = uidpair2kpair(cur_uids_K, uidA, uidB)
    except IndexError as e:
        # Move not allowed. Send out skip signal.
        print 'index error ....'
        from IPython import embed; embed()
        return None, None, None
    prop_uids_K = np.delete(cur_uids_K, kB)

    # Merge update statistics
    prop_SS_update = dict()
    for key, cur_arr in SS_update.items():
        if cur_arr.shape[0] == K:
            prop_arr = np.delete(cur_arr, kB, axis=0)
            prop_arr[kA] += cur_arr[kB]
            prop_SS_update[key] = prop_arr
        else:
            prop_SS_update[key] = cur_arr

    # Merge loss statistics
    prop_SS_loss = dict()
    prop_SS_loss['H_resp_K'] = np.delete(SS_loss['H_resp_K'], kB)
    prop_SS_loss['H_resp_K'][kA] = seed_SS_loss['H_resp_kA']
    return prop_SS_update, prop_SS_loss, prop_uids_K

def calc_local_params_for_merge_proposal(
        data, LP,
        GP=None,
        HP=None,
        cur_uids_K=None,
        uid_pair=None,
        **m_kwargs):
    ''' Construct proposed local parameters for given merge proposal

    Returns
    -------
    prop_LP : dict

    Examples
    --------
    >>> prng = np.random.RandomState(0)
    >>> LP = dict(resp_NK=prng.dirichlet(np.ones(4), size=5))
    >>> prop_LP = calc_local_params_for_merge_proposal(
    ...     None, LP, uid_pair=(0,1))
    >>> assert prop_LP['resp_NK'].shape[1] == 3
    >>> assert np.allclose(1, np.sum(prop_LP['resp_NK'], axis=1))
    '''
    if cur_uids_K is None:
        K = LP['resp_NK'].shape[1]
        cur_uids_K = np.arange(K)
    uidA, uidB = uid_pair
    kA, kB = uidpair2kpair(cur_uids_K, uidA, uidB)

    prop_resp_NK = LP['resp_NK'].copy()
    prop_resp_NK[:, kA] += prop_resp_NK[:, kB]
    prop_resp_NK = np.delete(prop_resp_NK, kB, axis=1)
    cur_uids_K = np.delete(cur_uids_K, kB)
    return dict(resp_NK=prop_resp_NK)


def calc_prop_H_resp_kA(resp_NK, kA, kB):
    K = resp_NK.shape[1]
    tmp_N = resp_NK[:, kA] + resp_NK[:, kB]
    tmp_N *= np.log(tmp_N)
    prop_H_resp_kA = -1 * np.sum(tmp_N, axis=0)
    return prop_H_resp_kA
