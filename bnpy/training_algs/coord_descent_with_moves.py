import numpy as np

import bnpy.data
from bnpy.alloc_models.mixtures \
    import dp_mix_vb_posterior_estimator as amod
from bnpy.obs_models \
    import gauss_diag_covar_vb_posterior_estimator as omod
from bnpy.alloc_models \
    import hierarchical_model as hmod
from bnpy.utils_io import parse_user_input_into_kwarg_dict

default_alg_kwargs = dict(
    move_names=None,
    n_laps=1,
    do_loss_after_local=False,
    do_loss_after_global=True,
    )


def make_uid_generator(start, max_n_iters=1000000):
    for i in range(1, max_n_iters):
        yield start + i

def fit(
        mod_list, data, GP, HP,
        uids_K=None,
        local_step_kwargs=None,
        global_step_kwargs=None,
        move_plan_kwargs=None,
        move_eval_kwargs=None,
        move_names=default_alg_kwargs['move_names'],
        n_laps=default_alg_kwargs['n_laps'],
        do_loss_after_local=default_alg_kwargs['do_loss_after_local'],
        do_loss_after_global=default_alg_kwargs['do_loss_after_global'],
        **alg_kwargs):
    ''' Run full-dataset coordinate descent training on specific dataset

    Returns
    -------
    GP : dict, updated global params
    '''
    if move_names is None:
        move_names = list()
    elif isinstance(move_names, str):
        move_names = move_names.split(',')

    if uids_K is None:
        K = GP[GP.keys()[0]].shape[0]
        uids_K = np.arange(K)
    uid_generator = make_uid_generator(K)

    # Parse input
    if local_step_kwargs is None:
        local_step_kwargs = dict()
    if global_step_kwargs is None:
        global_step_kwargs = dict()
    if move_plan_kwargs is None:
        move_plan_kwargs = dict()
    if move_eval_kwargs is None:
        move_eval_kwargs = dict()


    loss_history = list()
    LP = None
    SS_update = None
    SS_loss = None
    for lap_id in range(n_laps):
        plan_dict_list = hmod.make_plans_for_proposals(
            mod_list, data, LP, SS_update, SS_loss, GP, HP,
            move_names=move_names,
            cur_uids_K=uids_K,
            uid_generator=uid_generator,
            **move_plan_kwargs)

        # LOCAL STEP
        LP = hmod.calc_local_params(
            mod_list, data, GP, HP, **local_step_kwargs)

        # SUMMARY STEP
        SS_update = hmod.summarize_local_params_for_update(
            mod_list, data, LP)
        SS_loss = hmod.summarize_local_params_for_loss(
            mod_list, data, LP)

        # Make proposed summary "seeds"
        seed_list = hmod.make_summary_seeds_for_proposals(
            mod_list, data, LP, SS_update, SS_loss, GP, HP,
            plan_dict_list=plan_dict_list,
            cur_uids_K=uids_K,
            local_step_kwargs=local_step_kwargs,
            **move_plan_kwargs)

        # Update global params
        GP = hmod.update_global_params_from_summaries(
            mod_list, SS_update, GP, HP)
        loss = hmod.calc_loss_from_summaries(
            mod_list, GP, HP, SS_update, SS_loss)
        print loss
        print SS_update['n_K']
        # Evaluate any proposals
        GP, SS_update, SS_loss, loss, uids_K = hmod.evaluate_proposals(
            mod_list, GP, SS_update, SS_loss, HP,
            seed_list=seed_list,
            cur_loss=loss,
            cur_uids_K=uids_K,
            **move_eval_kwargs)
        print loss
        print SS_update['n_K']

        loss_history.append(loss)

    return GP, dict(
        SS_update=SS_update,
        SS_loss=SS_loss,
        uids_K=uids_K,
        loss_history=loss_history)

if __name__ == '__main__':
    data, mod_list, _, kwargs = parse_user_input_into_kwarg_dict()

    if data is None:
        prng = np.random.RandomState(0)
        X_a_NaD = 0.1 * prng.randn(500, 3) - 5
        X_b_NbD = 0.1 * prng.randn(500, 3) + 0
        X_c_NcD = 0.1 * prng.randn(5, 3) + 5
        X_ND = np.vstack([X_a_NaD, X_b_NbD, X_c_NcD])

    for key in kwargs:
        print key, kwargs[key]
    GP, HP, _, local_step_kwargs, extra_kwargs = \
        hmod.create_and_initialize_hierarchical_model_for_dataset(
            mod_list, X_ND, **kwargs)

    data = bnpy.data.XData(X_ND)
    fit(mod_list, data, GP, HP,
        local_step_kwargs=local_step_kwargs,
        move_plan_kwargs=dict(
            s_Knew=3),
        **extra_kwargs)
    
    """
    import bnpy.data
    from bnpy.alloc_models.mixtures.dp_mix_vb_split_proposals import \
        calc_local_params_for_split_proposal, \
        calc_seed_summaries_for_split_proposals, \
        make_full_summaries_from_seed_for_split_proposal
    data = bnpy.data.XData(X_ND)
    fdict = hmod.make_function_dict(mod_list)
    LP = fdict['calc_local_params'](data, GP, HP, **local_step_kwargs)

    '''
    s_kwargs = kwargs.copy()
    s_kwargs.update(fdict)
    prop_LP, prop_uids = calc_local_params_for_split_proposal(
        data, 
        LP=LP,
        GP=GP,
        HP=HP,
        uid=0,
        new_uids=[44,45,46],
        local_step_kwargs=local_step_kwargs,
        **s_kwargs)
    '''

    plan_dict_list = [
        dict(uid=0, new_uids=[-1, -2, -3, -4]),
        ]
    s_kwargs = kwargs.copy()
    s_kwargs.update(fdict)
    seed_dict_list = calc_seed_summaries_for_split_proposals(
        data=data, 
        LP=LP,
        GP=GP,
        HP=HP,
        plan_dict_list=plan_dict_list,
        local_step_kwargs=local_step_kwargs,
        **s_kwargs)
    """