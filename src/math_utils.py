import numpy as np
import itertools

def round(y, t='def', seed=10):
    assert t in ['def', 'bernulli']
    x = np.asarray(y)
    prng = np.random.RandomState(seed)

    if t == 'def':
        return np.round(x)
    else:
        p = np.ceil(x)-x
        return np.floor(x) + prng.binomial(n=1, size=x.shape, p=p)
        # a = prng.binomial(n=1, size=x.size, p=np.ravel(p)).reshape(x.shape)
        # return np.floor(x) + a#prng.binomial(n=1, size=x.size, p=p).reshape(x.shape)

def discrete_laplace(loc, scale, size=1, prng=np.random):
    # Following https://www.sciencedirect.com/science/article/pii/S0378375804003519
    # and https://rdrr.io/cran/extraDistr/man/DiscreteLaplace.html
    N = 10*(2*scale**2)     # ten times the variance
    p = np.exp(-1/scale)  # pr. of discrete Laplace, given the scale
    x = np.arange(loc-N, loc+N, 1)
    pr = np.abs((1-p)/(1+p) * p**(np.abs(x - loc)))
    pr[int(N/2)] += (1-sum(pr))
    return prng.choice(x, p=pr, size=size).astype(int)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def normalize(x):
    return x / np.sum(x)

def relu(x, inplace=True):
    ''' max(x, 0)'''
    if inplace:
        np.maximum(x, 0, x)
        return x
    else:
        return np.maximum(x, 0)

def div(x, y):
    """Ensures that numpy performs integer division if array in
       denominator has integer type.
    """
    if type(y) is np.ndarray and len(y.shape) == 1 and issubclass(y.dtype.type, np.integer):
        return x // y
    else:
        return x / y

def accumulate(x):
    ''' Returns the cumulative counts of a vector '''
    return np.asanyarray(list(itertools.accumulate(x)))

def rev_accumulate(x):
    ''' Inverse cumulative count operation '''
    ret = np.zeros(len(x), dtype=int)
    ret[0] = x[0]
    for i in range(len(x) - 1):
        ret[i + 1] = x[i + 1] - x[i]
    return ret
