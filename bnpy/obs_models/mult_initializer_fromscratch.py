import numpy as np

def init_global_params():
    '''
    '''
    pass

def init_local_params(
        dataset_or_X,
        K=5,
        seed=42,
        init_procedure='LP_from_rand_examples',
        **kwargs):
    ''' Initialize local responsibilities via data-driven procedure

    Returns
    -------
    LP : dict with arrays
        * resp_NK : 2D array, size N x K
        OR
        * resp_UK : 2D array, size U x K, U = n_unique_tokens
    info_dict : dict with info about how LP was generated

    Examples
    --------
    >>> X_NV = np.eye(3) + np.ones((3,3))
    >>> LP, _ = init_local_params(
    ...     X_NV, K=2, seed=0, init_procedure='LP_from_rand_examples')
    >>> LP['resp_NK']
    array([[ 0.,  0.],
           [ 0.,  1.],
           [ 1.,  0.]])
    '''
    # Parse raw data array from user input
    if isinstance(dataset_or_X, np.ndarray):
        X_NV = np.asarray(dataset_or_X)
    else:
        X_NV = dataset_or_X.X
    N, V = X_NV.shape

    # Pseudo-random generator
    prng = np.random.RandomState(seed)
    resp_NK = np.zeros((N, K))
    if init_procedure == 'LP_from_rand_examples':
        # Choose K items uniformly at random
        chosen_ids = prng.choice(np.arange(N), K, replace=False)
        for k in xrange(K):
            resp_NK[chosen_ids[k], k] = 1.0

    elif init_procedure == 'LP_from_rand_examples_by_dist':
        # Choose K items from the Data at random
        # Weighting choices by Euclidean distance
        # TODO: Change to KL distance??
        objID = prng.choice(N)
        chosen_ids = [objID] 
        min_dist_N = np.inf * np.ones(N)
        for k in range(1, K):
            tmp_ND = X_NV - X_NV[objID]
            np.square(tmp_ND, out=tmp_ND)
            cur_dist_N = np.sum(tmp_ND, axis=1)
            min_dist_N = np.minimum(min_dist_N, cur_dist_N)
            objID = prng.choice(
                N, p=min_dist_N / min_dist_N.sum())
            chosen_ids.append(objID)
        for k in xrange(K):
            resp_NK[chosen_ids[k], k] = 1.0

    return dict(resp_NK=resp_NK), dict(chosen_ids=chosen_ids)