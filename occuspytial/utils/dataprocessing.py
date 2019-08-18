from functools import lru_cache
from operator import itemgetter
from typing import Iterable

import numpy as np


class DetectionDataObject:

    def __init__(self, data):
        self.data = data

    @lru_cache(maxsize=None)
    def __getitem__(self, args):

        if isinstance(args, int):
            return self.data[args]
        if isinstance(args, Iterable):
            out = itemgetter(*args)(self.data)
            return np.concatenate(out)

    def __len__(self):
        return len(self.data)

    @property
    def visits_per_site(self):

        surveyed_sites = range(len(self))
        visits = tuple(self.data[i].shape[0] for i in surveyed_sites)
        return np.array(visits, dtype=np.int64)
