import numpy as np

from bnpy.obs_models \
    import gauss_diag_covar_vb_posterior_estimator as obs_mod

def fit(data, obs_mod, PD, HD, n_iters=10, **kwargs):
    loss_history = list()
    for i in range(n_iters):
        # LOCAL STEP
        log_proba_NK = obs_mod.calc_log_proba(data, PD)
        log_proba_NK -= log_proba_NK.max(axis=1)[:,np.newaxis]

        # Artificially enforce L-sparsity of 1
        # Meaning each data assigned to one "closest" cluster
        log_proba_NK[log_proba_NK != 0] = -1000
        np.exp(log_proba_NK, out=log_proba_NK)
        resp_NK = log_proba_NK
        resp_NK /= resp_NK.sum(axis=1)[:,np.newaxis]
        LP = dict(resp_NK=resp_NK)

        # Compute summary stats
        SS = obs_mod.summarize_local_params_for_update(data, LP)
        loss_v1 = obs_mod.calc_loss_from_summaries(SS, PD, HD)
        loss_history.append(loss_v1)
        print "% .5e" % loss_v1

        # Update global params
        PD = obs_mod.update_global_params_from_summaries(SS, PD, HD)
        loss_v2 = obs_mod.calc_loss_from_summaries(SS, PD, HD)
        loss_history.append(loss_v2)

        print "% .5e" % loss_v2
    # Verify monotonic decrease
    assert np.all(np.sign(np.diff(loss_history)) <= 0)
    return PD, dict(loss_history=loss_history)

if __name__ == '__main__':
    prng = np.random.RandomState(0)
    X_a_NaD = 0.1 * prng.randn(500, 3) - 5
    X_b_NbD = 0.1 * prng.randn(500, 3) + 0
    X_c_NcD = 0.1 * prng.randn(5, 3) + 5
    X_ND = np.vstack([X_a_NaD, X_b_NbD, X_c_NcD])
    hyper_dict = obs_mod.init_hyper_params(X_ND, prior_covar_x='0.1*eye')
    PD, _ = obs_mod.init_global_params(
        X_ND, hyper_dict, K=3,
        #init_procedure='LP_from_rand_examples',
        init_procedure='LP_from_rand_examples_by_dist',
        )
    opt_PD, info_dict = fit(X_ND, obs_mod, PD, hyper_dict)