import numpy as np
import sys
import os
import errno

def create_dir(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

def write(file, content, mode='a'):
    with open(file, mode) as myfile:
        myfile.write(content)

def get_and_slice(array: np.ndarray, l: list):
    """ Returns a slice of an array given the list of elements in l """
    if not l:
        return array
    l, i = l[1:], l[0]
    return get_and_slice(array[i], l)


import time
from functools import wraps

PROF_DATA = {}

def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time

        if fn.__name__ not in PROF_DATA:
            PROF_DATA[fn.__name__] = [0, []]
        PROF_DATA[fn.__name__][0] += 1
        PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling

def print_prof_data():
    for fname, data in PROF_DATA.items():
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        print("Function %s called %d times. " % (fname, data[0]),
              '\tExecution time max: %.3f, average: %.3f' % (max_time, avg_time))

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}