import numpy as np
import itertools

def calc_bar_z_K(resp_UK=None, wc_U=None):
    ''' Compute empirical doc-topic probability vector

    Returns
    -------
    barz_K : 1D array, size K
        barz_K[k] = fraction of tokens in doc explained by topic k
    '''
    return np.dot(wc_U, resp_UK) / np.sum(wc_U)

def calc_bar_z_outer_prod_KK(resp_UK=None, wc_U=None):
    ''' Compute outer-product of empirical doc-topic probability vector

    Uses smarter for-loop strategy that uses U^2/2 outer product calculations

    Returns
    -------
    barz_KK : 2D array, size K x K
        barz_KK[j, k] = frac of pairs of tokens explained by j and k
            = [ \sum_{u} \sum_{v} c_u r_uj c_v r_vk ] / (sum(c)*sum(c))
    '''
    U, K = resp_UK.shape
    # Element-wise multiply
    wresp_UK = wc_U[:, None] * resp_UK
    barz_KK = np.zeros((K, K))
    # Loop over pairs of rows (u,v) where u == v
    for u in xrange(U):
        barz_KK += (wc_U[u] **2) * np.diag(resp_UK[u])
    # Loop over pairs of rows (u,v) where u != v
    for u1, u2 in itertools.combinations(xrange(U), 2):
        tmp_KK = np.outer(wresp_UK[u1], wresp_UK[u2])
        barz_KK += tmp_KK
        barz_KK += tmp_KK.T
    return barz_KK / (np.sum(wc_U)**2)

def calc_bar_z_outer_prod_KK__slow(resp_UK=None, wc_U=None):
    ''' Compute outer-product of empirical doc-topic probability vector

    Uses naive for-loop strategy over all pairs of rows
    which requires U^2 outer product calculations

    Returns
    -------
    barz_KK : 2D array, size K x K
        barz_KK[j, k] = frac of pairs of tokens explained by j and k
            = [  \sum_{u \neq v} c_u r_uj c_v r_vk
               + \sum_{u} c_u^2 r_uj               ] / (sum(c)*sum(c))
    '''
    U, K = resp_UK.shape
    # Element-wise multiply
    wresp_UK = wc_U[:, None] * resp_UK

    # Loop over pairs of rows of resp
    # e.g. if U = 3, the we'll iterate over
    # 0,0 & 0,1 & 0,2 & 1,0 & 1,1 & 1,2 & 2,0 & 2,1 & 2,2
    barz_KK = np.zeros((K, K))
    for u1, u2 in itertools.product(xrange(U), xrange(U)):
        if u1 == u2:
            barz_KK += (wc_U[u1] **2) * np.diag(resp_UK[u1])
        else:
            barz_KK += np.outer(wresp_UK[u1], wresp_UK[u2])
    return barz_KK / (np.sum(wc_U)**2)

if __name__ == '__main__':
    U = 11 # unique word types in the doc
    K = 3  # num topics
    prng = np.random.RandomState(31)

    # Generate random approximate posterior parameter "resp_UK"
    resp_UK = prng.rand(U, K)**2
    resp_UK /= resp_UK.sum(axis=1)[:,np.newaxis]

    # Generate random word counts
    wc_U = prng.randint(low=1, high=3, size=U)

    print 'E[ barz * barz^T ] via formula'
    formula_barz_KK = calc_bar_z_outer_prod_KK(resp_UK, wc_U)
    print formula_barz_KK

    # Verify that our faster calculation is correct
    assert np.allclose(
        calc_bar_z_outer_prod_KK(resp_UK, wc_U),
        calc_bar_z_outer_prod_KK__slow(resp_UK, wc_U))

    # Verify by compute outer product via Monte Carlo expectation
    barz_KK_list = list()
    for samp_iter in xrange(50000):
        # Sample a z_U vector
        z_UK = np.zeros((U,K))
        for u in xrange(U):
            k = prng.choice(K, p=resp_UK[u])
            z_UK[u, k] = 1.0
        # Compute barz 1d array
        barz_K = np.dot(wc_U, z_UK) / np.sum(wc_U)
        # Compute outer product
        barz_KK = np.outer(barz_K, barz_K)
        barz_KK_list.append(barz_KK)
    mc_barz_KK = np.mean(np.dstack(barz_KK_list), axis=2)
    print 'E[ barz * barz^T ] via MonteCarlo estimator'
    print mc_barz_KK    
