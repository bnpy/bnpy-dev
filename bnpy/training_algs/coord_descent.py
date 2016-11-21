import numpy as np

from bnpy.alloc_models.mixtures \
    import dp_mix_vb_posterior_estimator as amod
from bnpy.obs_models \
    import gauss_diag_covar_vb_posterior_estimator as omod
from bnpy.alloc_models \
    import hierarchical_model as hmod

from bnpy.utils_io import parse_user_input_into_kwarg_dict

default_alg_kwargs = dict(
    n_laps=1,
    do_loss_after_local=True,
    do_loss_after_global=True,
    )

def fit(
        mod_list, data, GP, HP,
        local_step_kwargs=None, global_step_kwargs=None,
        n_laps=default_alg_kwargs['n_laps'],
        do_loss_after_local=default_alg_kwargs['do_loss_after_local'],
        do_loss_after_global=default_alg_kwargs['do_loss_after_global'],
        **alg_kwargs):
    ''' Run full-dataset coordinate descent training on specific dataset

    Returns
    -------
    GP : dict, updated global params
    '''
    # Parse input
    if local_step_kwargs is None:
        local_step_kwargs = dict()
    if global_step_kwargs is None:
        global_step_kwargs = dict()
    loss_history = list()
    for lap_id in range(n_laps):
        # LOCAL STEP
        LP = hmod.calc_local_params(
            mod_list, data, GP, HP, **local_step_kwargs)

        # SUMMARY STEP
        SS = hmod.summarize_local_params_for_update(
            mod_list, data, LP)
        SS_for_loss = hmod.summarize_local_params_for_loss(
            mod_list, data, LP)

        # Update global params
        GP = hmod.update_global_params_from_summaries(
            mod_list, SS, GP, HP)

        if do_loss_after_global:
            loss = hmod.calc_loss_from_summaries(
                mod_list, GP, HP, SS, SS_for_loss)
            loss_history.append(loss)
            print "% .5e" % loss

    return GP, dict(
        SS=SS,
        SS_for_loss=SS_for_loss,
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
    GP, HP, _, local_kwargs = \
        hmod.create_and_initialize_hierarchical_model_for_dataset(
            mod_list, X_ND, **kwargs)
    fit(mod_list, X_ND, GP, HP, local_kwargs=local_kwargs, n_laps=10)