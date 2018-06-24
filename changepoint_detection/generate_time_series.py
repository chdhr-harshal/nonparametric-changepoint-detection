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

def generate_motivating_example(minl=50, maxl=100):
    """Generate a motivating example time series.

    Args:
        minl (int): Minimum length of partition
        maxl (int): Maximum length of partition

    Returns:
        partition (int): Number of partitions
        data (np.array): Numpy array of data points
    """
    dim = 2
    num = 3
    partition = np.random.randint(minl, maxl, num)
    mu = np.zeros(dim)
    sigma1 = np.asarray([[1.0,0.75],[0.75,1.0]])
    data = np.random.multivariate_normal(mu, sigma1, partition[0])
    sigma2 = np.asarray([[1.0,0.0],[0.0,1.0]])
    data = np.concatenate((data, np.random.multivariate_normal(mu, sigma2, partition[1])))
    sigma3 = np.asarray([[1.0,-0.75],[-0.75,1.0]])
    data = np.concatenate((data, np.random.multivariate_normal(mu, sigma3, partition[2])))
    return partition, data

