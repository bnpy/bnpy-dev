import numpy as np
import copy

from bnpy.alloc_models.mixtures \
    import dp_mix_vb_posterior_estimator as amod
from bnpy.obs_models \
    import gauss_diag_covar_vb_posterior_estimator as omod
from bnpy.alloc_models \
    import hierarchical_model as hmod
from bnpy.data import fetch_batch
from bnpy.utils_io import parse_user_input_into_kwarg_dict

default_alg_kwargs = dict(
    n_laps=1,
    n_batches=1,
    do_loss_after_local=True,
    do_loss_after_global=True,
    seed=8675309,
    )

def fit(
        mod_list,
        dataset,
        GP, HP,
        local_step_kwargs=None,
        global_step_kwargs=None,
        n_laps=default_alg_kwargs['n_laps'],
        n_batches=default_alg_kwargs['n_batches'],
        do_loss_after_local=default_alg_kwargs['do_loss_after_local'],
        do_loss_after_global=default_alg_kwargs['do_loss_after_global'],
        seed=default_alg_kwargs['seed'],
        **alg_kwargs):
    '''

    Returns
    -------
    GP : dict, updated global params
    info_dict : dict of information about this run
    '''
    # Parse input
    if local_step_kwargs is None:
        local_step_kwargs = dict()
    if global_step_kwargs is None:
        global_step_kwargs = dict()

    prng = np.random.RandomState(seed)
    memorized_SS_per_batch = dict()
    SS_update = None
    SS_loss = None
    loss_history = list()

    for lap_id in range(n_laps):
        prng = np.random.RandomState(int(seed + lap_id))
        batch_list_for_cur_lap = prng.permutation(n_batches)
        for pos, batch_id in enumerate(batch_list_for_cur_lap):
            lap_frac = lap_id + pos / float(n_batches)
            batch_data = fetch_batch(dataset, batch_id)

            # CURRENT BATCH LOCAL STEP
            batch_LP = hmod.calc_local_params(
                mod_list, batch_data, GP, HP, **local_step_kwargs)

            # CURRENT BATCH SUMMARY STEP
            batch_SS_update = hmod.summarize_local_params_for_update(
                mod_list, batch_data, batch_LP)
            batch_SS_loss = hmod.summarize_local_params_for_loss(
                mod_list, batch_data, batch_LP)

            # TOTAL SUMMARY STEP
            if batch_id in memorized_SS_per_batch:
                old_batch_SS_update, old_batch_SS_loss = \
                    memorized_SS_per_batch[batch_id]
                SS_update = decrement_in_place(SS_update, old_batch_SS_update)
                SS_loss = decrement_in_place(
                    SS_loss, old_batch_SS_loss)
            SS_update = increment_in_place(SS_update, batch_SS_update)
            SS_loss = increment_in_place(SS_loss, batch_SS_loss)
            memorized_SS_per_batch[batch_id] = (
                batch_SS_update, batch_SS_loss)

            # Update global params
            GP = hmod.update_global_params_from_summaries(
                mod_list, SS_update, GP, HP)

            if do_loss_after_global:
                loss = hmod.calc_loss_from_summaries(
                    mod_list, GP, HP, SS_update, SS_loss)
                loss_history.append(loss)
                print "% .5e" % loss

    return GP, dict(
        SS_update=SS_update,
        SS_loss=SS_loss,
        loss_history=loss_history)

def decrement_in_place(SS, batch_SS):
    ''' Subtract summaries of current batch from whole-dataset summaries

    Does not allocate any new memory

    Returns
    -------
    SS : dict, same array fields as SS

    Examples
    --------
    >>> SS = dict(n_docs=10, n_K=np.asarray([100., 200., 300.]))
    >>> batch_SS = dict(n_docs=1, n_K=np.asarray([4., 5., 6.]))
    >>> out_SS = decrement_in_place(SS, batch_SS)
    >>> assert out_SS['n_docs'] == 9
    >>> assert np.allclose(out_SS['n_K'], [96, 195, 294])
    '''
    for key, arr in SS.items():
        arr -= batch_SS[key]
        SS[key] = arr
    return SS

def increment_in_place(SS, batch_SS):
    ''' Add summaries from current batch to whole-dataset summaries

    Does not allocate any new memory

    Returns
    -------
    SS : dict, same array fields as SS

    Examples
    --------
    >>> SS = dict(n_docs=10, n_K=np.asarray([100., 200., 300.]))
    >>> batch_SS = dict(n_docs=1, n_K=np.asarray([4., 5., 6.]))
    >>> out_SS = increment_in_place(SS, batch_SS)
    >>> assert out_SS['n_docs'] == 11
    >>> assert np.allclose(out_SS['n_K'], [104, 205, 306])
    >>> assert SS['n_docs'] == 11

    # Verify base case, where input is None
    >>> SS = increment_in_place(None, batch_SS)
    >>> assert SS['n_docs'] == batch_SS['n_docs']
    '''
    if SS is None:
        return copy.deepcopy(batch_SS)
    for key, arr in SS.items():
        arr += batch_SS[key]
        SS[key] = arr
    return SS

if __name__ == '__main__':
    data, mod_list, _, user_input_kwargs = parse_user_input_into_kwarg_dict()

    if data is None:
        prng = np.random.RandomState(0)
        X_a_NaD = 0.1 * prng.randn(500, 3) - 5
        X_b_NbD = 0.1 * prng.randn(500, 3) + 0
        X_c_NcD = 0.1 * prng.randn(5, 3) + 5
        X_ND = np.vstack([X_a_NaD, X_b_NbD, X_c_NcD])

    GP, HP, _, local_kwargs = \
        hmod.create_and_initialize_hierarchical_model_for_dataset(
            mod_list, X_ND,
            **user_input_kwargs)
    fit(mod_list, X_ND, GP, HP,
        local_kwargs=local_kwargs,
        **user_input_kwargs)