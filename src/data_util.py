import numpy as np
import itertools

def shave(y):
    _y = []
    for i, c in enumerate(y):
        _y += list(range(c))
    return np.asarray(_y)

def select(y):
    unique, counts = np.unique(y, return_counts=True)
    return counts

def accumulate(x):
    ''' Returns the cumulative counts of a vector '''
    return np.asanyarray(list(itertools.accumulate(x)))

def revAccumulate(x):
    ''' Inverse cumulative count operation '''
    ret = np.zeros(len(x), dtype=int)
    ret[0] = x[0]
    for i in range(len(x) - 1):
        ret[i + 1] = x[i + 1] - x[i]
    return ret

def get_shape(df):
    ''''Returns the shape of a pandas dataframe'''
    s = df.shape
    return (s[0], 1) if len(s) == 1 else s

