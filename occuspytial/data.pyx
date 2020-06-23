# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
from numbers import Number
from typing import Iterable

import numpy as np


cdef class Data:

    cdef public data
    cdef public surveyed
    cdef dict _cache

    def __init__(self, data):
        self.data = data
        self.surveyed = list(data)
        self._cache = {}

    def visits(self, sites):
        """get visits per site for input sites"""
        cdef Py_ssize_t i
        cdef int size = 0

        if isinstance(sites, Number):
            out_num = self.data[sites].shape[0]
            return out_num
        if isinstance(sites, Iterable):
            out_list = []
            size = sites.shape[0]
            for i in range(size):
                j = sites[i]
                visits = self.data[j].shape[0]
                out_list.append(visits)
            return out_list

    def __getitem__(self, sites):
        """Get concatenated value(s) of ``self.data`` corresponding to
        the key(s) provided in ``sites``.
        """
        cdef Py_ssize_t i, j
        cdef int size = 0
 
        try:
            try:
                key = tuple(sites)
                size = len(key)
            except TypeError:
                # when sites is an integer
                key = (sites,)
                size = 1
            out = self._cache[key]
        except KeyError:
            group = []
            for i in range(size):
                j = key[i]
                value = self.data[j]
                group.append(value)
            out = np.concatenate(group)
            self._cache[key] = out
        return out

    def __len__(self):
        return len(self.data)
