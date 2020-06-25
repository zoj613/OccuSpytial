from numbers import Number
from operator import itemgetter
from typing import Iterable

import numpy as np


class Data:

    def __init__(self, data):
        self.data = data
        self.surveyed = list(data)

    def visits(self, sites):
        """get visits per site for input sites"""
        if isinstance(sites, Number):
            out = self.data[sites].shape[0]
        if isinstance(sites, Iterable):
            out = tuple(j.shape[0] for j in itemgetter(*sites)(self.data))
        return out

    def __getitem__(self, sites):
        try:
            out = itemgetter(*sites)(self.data)
            out = np.concatenate(out)
        except TypeError:
            out = self.data[sites]
        return out

    def __len__(self):
        return len(self.data)
