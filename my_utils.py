import os
import numpy as np


def cache(f, path):
    def cached(*args, **kwargs):
        if os.path.exists(path):
            data = np.load(path)
            return data
        else:
            data = f(*args, **kwargs)
            np.savez(path, **data)
            return data
    return cached
