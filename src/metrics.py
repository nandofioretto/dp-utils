import numpy as np
from itertools import accumulate

class Metrics:
    @staticmethod
    def nonzero_err(x, xhat):
        return abs(np.count_nonzero(x.flatten()) - np.count_nonzero(xhat.flatten()))

    @staticmethod
    def sum_err(x, xhat):
        return abs(np.sum(x.flatten()) - np.sum(xhat.flatten()))

    @staticmethod
    def mean_err(x, xhat):
        return abs(np.mean(x.flatten()) - np.mean(xhat.flatten()))

    @staticmethod
    def std_err(x, xhat):
        return abs(np.std(x.flatten()) - np.std(xhat.flatten()))

    @staticmethod
    def median_err(x, xhat):
        return abs(np.median(x.flatten()) - np.median(xhat.flatten()))

    @staticmethod
    def relative_err(x, xhat, type=2):
        assert type in [1, 2]
        return np.mean(np.abs(x.flatten() - xhat.flatten())**type / (x.flatten() + 1))

    @staticmethod
    def norm(x, xhat, type=1):
        assert type in [1, 2, np.inf]
        norm = np.linalg.norm(x.flatten() - xhat.flatten(), type)
        return np.round(norm, 4)

    @staticmethod
    def normAvg(x, xhat, type=1):
        assert type in [1, 2, np.inf]
        norm = np.linalg.norm(x.flatten() - xhat.flatten(), type)
        return np.round(norm / float(len(x.flatten())), 4)

    @staticmethod
    def earthmoving_distance(x, tilde_x, transform=True):
        if transform:
            res = Metrics.norm(np.asarray(list(accumulate(x))), np.asarray(list(accumulate(tilde_x))))
        else:
            res = Metrics.norm(x, tilde_x)
        return np.round(res, 4)