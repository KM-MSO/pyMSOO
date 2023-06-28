import numpy as np
from numba import jit, njit
import math 
import scipy

@jit(nopython = True)
def numba_randomchoice_w_prob(a):
    assert np.abs(np.sum(a) - 1.0) < 1e-9
    rd = np.random.rand()
    res = 0
    sum_p = a[0]
    while rd > sum_p:
        res += 1
        sum_p += a[res]
    return res

@jit(nopython = True)
def numba_randomchoice(a, size= None, replace= True):
    return np.random.choice(a, size= size, replace= replace)

@jit(nopython= True) 
def numba_random_uniform(size = 1):
    return np.random.rand(size)

@jit(nopython= True) 
def numba_random_gauss(mean, sigma=0.1):
    return mean + sigma *math.sqrt(-2.0 * math.log(np.random.rand())) * math.sin(2.0 * math.pi * np.random.rand())

@jit(nopython=True)
def numba_random_cauchy(mean, sigma=0.1): 
    return np.random.standard_cauchy()*sigma + mean

@jit(nopython=True)
def numba_clip(array, low, high):
    return np.clip(array, low, high)

@jit(nopython = True)
def numba_linalgo_det(matrix):
    return np.linalg.det(matrix)

@jit(nopython = True)
def numba_linalgo_pinv(matrix):
    return np.linalg.pinv(matrix)

@jit(nopython = True)
def numba_dot(a, b):
    return np.dot(a, b)

@njit
def numba_argsort(arr):
    return np.argsort(np.array(arr))