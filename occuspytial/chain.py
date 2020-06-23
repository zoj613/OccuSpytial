import numpy as np


class Chain:
    def __init__(self, params, size):
        self.size = size
        self._names = tuple(params)
        self._store = dict.fromkeys(params)
        self._index = 0
        for key, value in self._store.items():
            cols = params[key]
            if cols > 1:
                array = np.zeros((size, cols))
            else:
                array = np.zeros(size)
            self._store[key] = array

    @property
    def full(self):
        out = []
        for val in self._store.values():
            if val.ndim > 1:
                out.append(val)
            else:
                out.append(val[:, None])

        return np.concatenate(out, axis=1)[:self._index]

    def append(self, params):
        if self._index > (self.size - 1):
            msg = 'Chain is full, cannot append any new values'
            raise ValueError(msg)
        for key, value in params.items():
            self._store[key][self._index] = value
        self._index += 1

    def expand(self, size):
        for key, value in self._store.items():
            dim = value.ndim
            if dim > 1:
                new = np.zeros((size, value.shape[1]))
                self._store[key] = np.append(value, new, axis=0)
            else:
                new = np.zeros(size)
                self._store[key] = np.append(value, new)
        self.size += size

    def __getitem__(self, name):
        return self._store[name][:self._index]

    def __len__(self):
        return self._index

    def __repr__(self):
        return f'Chain(params: {self._names}, size: {self._index})'
