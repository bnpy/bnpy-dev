import numpy as np
from scipy.special import digamma, gammaln

from calc_bar_z_for_slda import calc_bar_z_K, calc_bar_z_outer_prod_KK

def calc_local_params(data, LP,
        alpha_K=None,
        local_step_copy_large_arrays=False,
        local_step_min_resp_val=1e-100,
        local_step_resp_sum_N=1.0,
        **kwargs):
    ''' Compute local parameters for topic model

    Returns
    -------
    LP : dict with array fields
        * resp_NK
    '''
    # TODO update this
    return LP


def calc_local_params_for_single_doc__naive(
        wc_U=None,
        obs_log_proba_UK=None,
        alpha=None,
        alloc_proba_K=None,
        y=None,
        eta_K=None,
        delta=1.0,
        do_track_loss=0,
        local_step_seed=0,
        local_step_do_cold_start=True,
        local_step_conv_thr=0.01,
        local_step_max_iters=100,
        local_step_min_resp_val=1e-100,
        out_resp_UK=None,
        out_theta_K=None,
        out_count_K=None,
        ):
    ''' Estimate local parameters for single document

    Uses naive alg. explicitly calculates resp/theta at every step.

    Returns
    -------
    resp_UK : 2D array, size U x K
    theta_K : 1D array, size K
    count_K : 1D array, size K

    Examples
    --------
    >>> import time
    >>> prng = np.random.RandomState(int(time.time()))
    >>> U = 17
    >>> K = 7
    >>> alpha = 0.1 + prng.rand()
    >>> alloc_proba_K = np.ones(K) / float(K)
    >>> wc_U = prng.randint(low=1, high=10, size=U)
    >>> obs_log_proba_UK = np.log(prng.rand(U, K))

    >>> delta = 0.1 + prng.rand()
    >>> y = prng.randn()
    >>> eta_K = prng.randn(K)

    # Perform local step using a "cold" start
    >>> resp_UK, theta_K, count_K, info_dict = \\
    ...     calc_local_params_for_single_doc__naive(
    ...         alpha=alpha, alloc_proba_K=alloc_proba_K,
    ...         wc_U=wc_U, obs_log_proba_UK=obs_log_proba_UK,
    ...         eta_K=eta_K, y=y, delta=delta,
    ...         do_track_loss=1,
    ...         local_step_do_cold_start=True)    
    >>> np.allclose(1.0, np.sum(resp_UK, axis=1))
    True
    >>> np.allclose(wc_U.sum(), count_K.sum())
    True

    >>> #from matplotlib import pylab
    >>> #pylab.plot(info_dict['loss_history'])
    >>> #pylab.show()

    # Verify monotonic loss
    # Meaning differences are always negative (up to numerical precision)
    >>> np.diff(info_dict['loss_history']).max() < 1e-10
    True

    '''
    U, K = obs_log_proba_UK.shape
    N = np.sum(wc_U)

    # Allocate memory for returned arrays
    # or use provided "output" arrays for this purpose
    if out_theta_K is None:
        theta_K = np.zeros(K)
    else:
        theta_K = out_theta_K[:K]
    if out_count_K is None:
        count_K = np.zeros(K)
    else:
        count_K = out_count_K[:K]
    prevcount_K = -1 * np.ones(K)
    if out_resp_UK is None:
        resp_UK = np.zeros((U, K))
    else:
        resp_UK = resp_UK[:U, :K]

    info_dict = dict(
        local_step_do_cold_start=local_step_do_cold_start,
        do_track_loss=do_track_loss,
        )

    # Initialize
    # ----------
    if local_step_do_cold_start:
        # If desired, initialize resp parameters using cold start
        # log_resp \approx obs_log_proba + alloc_log_proba
        np.add(obs_log_proba_UK,  np.log(alloc_proba_K), out=resp_UK)
    else:
        # Otherwise, will use provided counts to initialize resp
        np.add(count_K, alpha * alloc_proba_K, out=theta_K)
        np.add(obs_log_proba_UK, digamma(theta_K), out=resp_UK)
    resp_UK -= np.max(resp_UK, axis=1)[:, np.newaxis]
    np.exp(resp_UK, out=resp_UK)
    resp_UK /= resp_UK.sum(axis=1)[:, np.newaxis]

    prng = np.random.RandomState(int(local_step_seed))
    # Refine
    # ------
    # Perform many iterations to refine these local parameters
    for riter in xrange(local_step_max_iters):
        for step_id, u in enumerate(prng.permutation(U)):
            # Update theta_K every few types
            if step_id % 5 == 0:
                np.dot(wc_U, resp_UK, out=count_K)
                np.add(count_K, alpha * alloc_proba_K, out=theta_K)

            # Update resp_K for current type u
            sumbarz_K = np.dot(wc_U, resp_UK) - wc_U[u] * resp_UK[u]
            resp_UK[u] = obs_log_proba_UK[u] + digamma(theta_K) + \
                0.5 / delta * (
                    + 2 * y * eta_K / N
                    - wc_U[u] * (eta_K * eta_K) / (N*N)
                    - 2 * np.inner(eta_K, sumbarz_K) * eta_K / (N*N))

            resp_UK[u] -= np.max(resp_UK[u])
            np.exp(resp_UK[u], out=resp_UK[u])
            resp_UK[u] /= resp_UK[u].sum()

            if do_track_loss and step_id % 5 == 0:
                cur_loss = calc_loss_single_doc(
                    resp_UK=resp_UK,
                    theta_K=theta_K,
                    alpha=alpha,
                    alloc_proba_K=alloc_proba_K,
                    wc_U=wc_U,
                    obs_log_proba_UK=obs_log_proba_UK,
                    y=y,
                    eta_K=eta_K,
                    delta=delta,
                    )
                if riter == 0:
                    loss_list = list()
                loss_list.append(cur_loss)

        if np.sum(np.abs(prevcount_K - count_K)) < .001:
            break
        prevcount_K[:] = count_K

    # Finalize
    # --------
    # So that output parameters meet expectations
    # That they are safe to take logarithms
    # And that counts/thetas have been updated directly from this resp
    np.maximum(resp_UK, local_step_min_resp_val, out=resp_UK)
    np.dot(wc_U, resp_UK, out=count_K)
    np.add(count_K, alpha * alloc_proba_K, out=theta_K)

    info_dict['max_iters'] = local_step_max_iters
    info_dict['n_iters'] = riter + 1
    if do_track_loss:
        info_dict['loss_history'] = np.asarray(loss_list)
    return resp_UK, theta_K, count_K, info_dict


def calc_loss_single_doc(
        resp_UK=None,
        theta_K=None,
        alpha=None,
        alloc_proba_K=None,
        wc_U=None,
        obs_log_proba_UK=None,
        delta=1.0,
        y=None,
        eta_K=None):
    ''' Compute loss (negative elbo) for single document's parameters

    Returns
    -------
    loss : scalar real
    '''
    wresp_UK = wc_U[:, np.newaxis] * resp_UK
    count_K = np.sum(wresp_UK, axis=0)

    # Compute theta terms
    L_theta_cumulant_dir = np.sum(gammaln(theta_K))
    L_theta_slack = np.inner(
        count_K + alpha * alloc_proba_K - theta_K,
        digamma(theta_K))
    # Compute resp terms
    L_resp_entropy = -1.0 * np.sum(wresp_UK * np.log(resp_UK))
    L_resp_obs = np.sum(wresp_UK * obs_log_proba_UK)
    # Compute final supervised term
    barz_K = calc_bar_z_K(resp_UK, wc_U)
    barz_KK = calc_bar_z_outer_prod_KK(resp_UK, wc_U)
    L_resp_super = -0.5 / delta * (
        - 2.0 * y * np.inner(eta_K, barz_K)
        + np.dot(eta_K, np.dot(barz_KK, eta_K)))

    L_single_doc = (
        L_resp_obs + L_resp_entropy + L_resp_super +
        L_theta_cumulant_dir + L_theta_slack)
    return -1.0 * L_single_doc