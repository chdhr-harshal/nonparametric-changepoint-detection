from __future__ import division
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def divisive(tdata, sig_lvl=0.05, R=199, k=None, min_size=50, alpha=1):
    """Main method to divide a timeseries into segments.

    Args:
        tdata (np.ndarray): Numpy array for a multivariate time series.
                            Each row is a new observation at different time.
        sig_lvl (double):   Significance level required from the permutation test.
        R (int):            Number of iterations in the permutation test.
        k (int):            Number of segments to divide a timeseries into (optional).
        min_size(int):      Minimum size of each segment.
        alpha (int):        alpha value in the divergence measure.

    Returns:
        sol (dict):         Dictionary containing the estimated changepoints and other
                            auxiliary information from the algorithm.
    """
    assert not(R < 0 and k is None), "R must be a non-negative integer."
    assert not((sig_lvl <=0 or sig_lvl >= 1) and k is None), "sig_lvl must be a positive real number between 0 and 1."
    assert not(min_size < 2), "min_size must be an integer greater than 1."
    assert not(alpha > 2 or alpha <= 0), "alpha must be in (0,2]."

    n = len(tdata)
    global energy
    energy = np.zeros(shape=(n,2), dtype=np.float64)

    # If k is provided, no need to perform a permutation test.
    if k is None:
        k = n
    else:
        R = 0 #no need to perform a permutation test

    changes = np.array([0,n])
    sol = dict({'k_hat' : 1})
    pvals = np.array([])
    permutations = np.array([])

    D = pairwise_distances(tdata)
    condition = None

    while k > 0:
        #Find the split point
        tmp = split(changes, D, min_size, False)
        i = tmp['first']
        j = tmp['second']
        Estat = tmp['fourth']
        tmp = tmp['third']

        con = tmp[-1]
        if con == -1:
            break

        result = significance_test(D, R, changes, min_size, Estat)
        pval = result[0]
        permutations = np.append(permutations, result[1])
        pvals = np.append(pvals, pval)

        if pval > sig_lvl:
            break

        changes = tmp
        sol['k_hat'] += 1
        k -= 1

    # Update the return dictionary
    tmp = np.sort(changes)
    sol['order_found'] = changes
    sol['estimates'] = tmp
    sol['considered_last'] = con
    sol['p_values'] = pvals
    sol['permutations'] = permutations
    sol['clusters'] = np.array([], dtype=np.int32)

    current_cluster = 0
    i = 0
    for changepoint in sol['estimates']:
        while i < changepoint:
            sol['clusters'] = np.append(sol['clusters'], current_cluster)
            i += 1
        current_cluster += 1

    print "Found {} split points".format(len(changes)-2)

    return sol

def split(changes, D, min_size, for_sim):
    """Split routine for the entire time series

    Args:
        changes (np.array):     Array with each element containing index
                                for the start of next segment. Last element of the
                                array is the last index of the timeseries.
        D (np.ndarray):         2-dimensional array containing pairwise distances
                                between observations at different times.
        min_size (int):         Minimum size of each segment.
        for_sim (bool):         True if the split routine is being called for significance testing.

    Returns:
        dict (dict):            Dictionary containing endpoints of the split and energy released
                                due to the split.
    """
    global energy
    splits = np.sort(changes)
    best = np.array([-1, -np.inf])
    ii = -1
    jj = -1
    if for_sim:
        # If the procedure is being used for significance testing
        for i in range(1,len(splits)):
            tmp = find_split_point(splits[i-1], splits[i]-1, D, min_size)
            if tmp[1] > best[1]: # tmp[1] is the energy released when the cluster is split
                ii = splits[i-1]
                jj = splits[i]-1
                best = tmp
        changes = np.append(changes, best[0])
        return {'first':ii, 'second':jj, 'third':changes, 'fourth':best[1]}
    else:
        for i in range(1,len(splits)):
            if energy[splits[i-1],0]:
                tmp = energy[splits[i-1],]
            else:
                tmp = find_split_point(splits[i-1], splits[i]-1, D, min_size)
                energy[splits[i-1], 0] = tmp[0]
                energy[splits[i-1], 1] = tmp[1]

            if tmp[1] > best[1]:
                ii = splits[i-1]
                jj = splits[i]-1
                best = tmp

        changes = np.append(changes, int(best[0]))
        energy[ii, 0] = 0
        energy[ii, 1] = 0
        return {'first':ii, 'second':jj, 'third':changes, 'fourth':best[1]}

def find_split_point(start, end, D, min_size):
    """Find splitpoint inside a given segment.

    Args:
        start (int):        Starting index of the segment
        end (int):          End index (last element) of the segment.
        D (np.ndarray):     Pairwise distance matrix
        min_size (int):     Minimum size of the segment.

    Returns:
        best (np.array):    Numpy array with first element as the split point.
                            Second element is the enery released by the split.
    """
    if end-start+1 < 2*min_size:
        return np.array([-1, -np.inf])

    D = D[start:end+1, start:end+1]

    total_points = end-start+1
    best = np.array([-1, -np.inf])

    t1 = min_size
    t2 = 2*t1

    cut1 = D[0:t1, 0:t1]
    cut2 = D[t1:t2, t1:t2]
    cut3 = D[0:t1, t1:t2]

    A = np.sum(cut1)/2
    B1 = np.sum(cut2)/2
    AB1 = np.sum(cut3)

    tmp = 2*AB1/((t2-t1)*t1) - 2*B1/((t2-t1-1)*(t2-t1)) - 2*A/((t1-1)*t1)
    tmp *= t1*(t2-t1)/t2

    if tmp > best[1]:
        best[0] = t1 + start
        best[1] = tmp


    t2 += 1

    B = np.full(total_points+1, B1)
    AB = np.full(total_points+1, AB1)

    while t2 <= total_points:
        B[t2] = B[t2-1] + np.sum(D[t2-1:t2, t1:t2-1])
        AB[t2] = AB[t2-1] + np.sum(D[t2-1:t2, 0:t1])

        tmp = 2*AB[t2]/((t2-t1)*t1) - 2*B[t2]/((t2-t1-1)*(t2-t1)) - 2*A/((t1-1)*t1)
        tmp *= t1*(t2-t1)/t2

        if tmp > best[1]:
            best[0] = t1 + start
            best[1] = tmp

        t2 += 1

    t1 += 1

    while True:
        t2 = t1 + min_size
        if t2 > total_points:
            break

        addA = np.sum(D[t1-1:t1, 0:t1-1])
        A += addA

        addB = np.sum(D[t1-1:t1, t1:t2-1])
        while t2 <= total_points:
            addB += D[t1-1,t2-1]
            B[t2] -= addB
            AB[t2] += (addB - addA)

            tmp = 2*AB[t2]/((t2-t1)*t1) - 2*B[t2]/((t2-t1-1)*(t2-t1)) - 2*A/((t1-1)*t1)
            tmp *= t1*(t2-t1)/t2

            if tmp > best[1]:
                best[0] = t1 + start
                best[1] = tmp

            t2 += 1
        t1 += 1

    return best

def significance_test(D, R, changes, min_size, obs):
    """Significance testing routine

    Args:
        D (np.ndarray):     Pairwise distance matrix.
        R (int):            Number of iterations in the significance test.
        min_size (int):     Minimum size of each segment.
        obs (double):       Energy released from the proposed split.

    Returns:
        np.array:           First element is p_val of the split.
                            Second element is the total iterations run.
    """
    if R == 0:
        return np.array([0, 0])
    over = 0
    for f in xrange(R):
        D1 = permute_within_cluster(D, changes)
        tmp = split(changes, D1, min_size, True)
        if tmp['fourth'] >= obs:
            over += 1

    p_val = (1 + over)/(f + 1)
    return np.array([p_val, f])

def permute_within_cluster(D, points):
    """Permutes the pairwise distance matrix for the permutation test.

    Args:
        D (np.ndarray):     Pairwise distance matrix.
        points (np.array):  The array containing split points.

    Returns:
        D1 (np.ndarray):    New permuted pairwise distance matrix.
    """
    D1 = np.copy(D)
    points = np.sort(points)
    K = len(points)-1
    for i in xrange(K): # Shuffle within clusters by permuting matrix columns and rows
        s_idx = points[i]
        e_idx = points[i+1]
        perm = np.random.permutation(e_idx-s_idx)
        D1[s_idx:e_idx,:] = np.take(D1[s_idx:e_idx,:],perm,0)
        D1[:,s_idx:e_idx] = np.take(D1[:,s_idx:e_idx],perm,1)
    return D1
