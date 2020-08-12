import numpy as np


class Chain:
    """Container to store parameter chains during sampling.

    Parameters
    ----------
    params : Dict[str, int]
        Dictionary of parameter metadata. The keys are the parameter names
        and the values are the number of dimensions of the parameter.
    size : int
        Length of the parameter chain.

    Attributes
    ----------
    full
    """

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
        """Return the full chain as a numpy array.

        The returned array is a concatenation of the arrays of all parameters.
        The number of columns is the sum of the parameters' dimensions. The
        number of rows may be less than `size` attribute if the chain is not
        full.

        Returns
        -------
        np.ndarray
            concatenated parameter chains.
        """
        out = []
        for val in self._store.values():
            if val.ndim > 1:
                out.append(val)
            else:
                out.append(val[:, None])

        return np.concatenate(out, axis=1)[:self._index]

    def append(self, params):
        """Append new values to the chain.

        Parameters
        ----------
        params : Dict[str, Union[float, nd.ndarray]]
            Dictionary of values to append. Keys are the parameter names.

        Raises
        ------
        ValueError
            If chain is already full (i.e. number of values per porameter is
            already equal to the `size` attribute.)
        """
        if self._index > (self.size - 1):
            raise ValueError('Chain is full, cannot append any new values')
        for key, value in params.items():
            self._store[key][self._index] = value
        self._index += 1

    def expand(self, size):
        """Extend the chain capacity by a specified length.

        Parameters
        ----------
        size : int
            Length to extend the chain by.
        """
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
        """Access individual parameter using its name.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        chain: np.ndarray
            Single parameter chain.
        """
        return self._store[name][:self._index]

    def __len__(self):
        """Return the size of the chain.

        This value is never greater than the chain's capacity `size`
        """
        return self._index

    def __repr__(self):
        return f'Chain(params: {self._names}, size: {self._index})'
