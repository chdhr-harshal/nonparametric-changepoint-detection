from __future__ import division
import numpy as np

def generate_univariate_time_series(num, minl=50, maxl=100):
    """Generate univariate normal time series.

    Args:
        num (int): Number of partitions
        minl (int): Minimum length of partition
        maxl (int): Maximum length of partition

    Returns:
        partition (int): Number of partitions
        data (np.array): Numpy array of data points
    """
    data = np.array([], dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.randn()*10
        std = np.random.randn()*1
        if std < 0:
            std *= -1
        tdata = np.random.normal(mean, std, p)
        data = np.concatenate((data, tdata))
    return partition, np.atleast_2d(data).T

def generate_multivariate_time_series(num, dim, minl=50, maxl=100):
    """Generate multivariate normal time series.

    Args:
        num (int): Number of partitions
        dim (int): Number of dimensions
        minl(int): Minimum length of partition
        maxl(int): Maximum length of partition

    Returns:
        partition (int): Number of partitions
        data (np.array): Numpy array of data points
    """
    data = np.empty((1,dim), dtype=np.float64)
    partition = np.random.randint(minl, maxl, num)
    for p in partition:
        mean = np.random.standard_normal(dim)*10
        # Generate a random SPD matrix
        A = np.random.standard_normal((dim,dim))
        var = np.dot(A, A.T)

        tdata = np.random.multivariate_normal(mean, var, p)
        data = np.concatenate((data, tdata))
    return partition, data[1:,:]
