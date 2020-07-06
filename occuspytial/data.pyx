#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True
from cpython.list cimport PyList_CheckExact
from cpython.object cimport PyObject, PyObject_GetItem
from cpython.ref cimport Py_INCREF, Py_XINCREF
from cpython.sequence cimport (
    PySequence_Fast_GET_SIZE,
    PySequence_Fast_GET_ITEM,
    PySequence_List,
)
from cpython.tuple cimport PyTuple_New, PyTuple_SET_ITEM, PyTuple_CheckExact
import numpy as np
cimport numpy as np

np.import_array()


# The following declarations are made to avoid explicit type casting inside the
# for-loop in the function ``_iterate_and_append``. Without these declarations
# and using those provided by Cython, explicity casting to <object> type would
# be necessary for some variables inside the loop.
cdef extern from 'Python.h':
    PyObject* PyDict_GetItem(PyObject* p, PyObject* key)
    Py_ssize_t PyDict_Size(PyObject* p)


cdef extern from 'numpy/arrayobject.h':
    int PyArray_DIM(PyObject* arr, int n) nogil


cdef class Data:
    """Container for Detection data.

    This class is most useful storing the conditional detection covariate
    data (``W``) and detection-survey data (``y``) in a more accessible way
    that allows convenient multiple site data access at once.

    Parameters
    ----------
    data : Dict[int, nd.ndarray]
        Dictionary of site data. The key should be the site numbers and
        value the relevant site data array (detection data or its design
        matrix).

    Attributes
    ----------
    surveyed : List[int]
        indices of sites were surveyed. This is calculated using `data` keys.

    Methods
    -------
    visits(sites)
    """
    cdef PyObject* data
    cdef object _pickle_data
    cdef public object surveyed

    def __cinit__(self, dict data):
        self.data = <PyObject*>data
        self._pickle_data = data
        self.surveyed = PySequence_List(data)

    cdef object _iterate_and_append(self, object obj, bint append_shape):
        """Iterate over `obj` and return a tuple with values of `data` whose
        keys is are elements of `obj`.
        """
        cdef:
            Py_ssize_t i, n = PySequence_Fast_GET_SIZE(obj)
            PyObject* data_item
            PyObject* item
            object array_shape, out

        out = PyTuple_New(n)
        for i in range(n):
            item = PySequence_Fast_GET_ITEM(obj, i)
            data_item = PyDict_GetItem(self.data, item)
            # *_SET_ITEM steals the argument's reference so we must increment
            # the reference count of `data_item` so the item of the `self.data`
            # dictionary does not get deallocated.
            if append_shape:
                array_shape = PyArray_DIM(data_item, 0)
                Py_INCREF(array_shape)
                PyTuple_SET_ITEM(out, i, array_shape)
            else:
                Py_XINCREF(data_item)
                PyTuple_SET_ITEM(out, i, <object>data_item)
        return out

    def visits(self, sites):
        """
        visits(sites)

        Return the number of visits per site.

        Parameters
        ----------
        sites : {int, List[int], Tuple[int]}
            Site(s) whose number of visits are to be returned.

        Returns
        -------
        out : {Tuple[int], int}
            Number of visits per site provided by `sites`.

        """
        if PyList_CheckExact(sites) | PyTuple_CheckExact(sites):
            out = self._iterate_and_append(sites, True)
        else:
            arr = PyDict_GetItem(self.data, <PyObject*>sites)
            out = PyArray_DIM(arr, 0)
        
        return out

    def __getitem__(self, sites):
        """Return data of a site.

        If sites is a sequence the returned data is a concatenated array of
        the data in all sites contained in `sites` along the first axis.

        Parameters
        ----------
        sites : {int, List[int], Tuple[int]}
            Site id/number(s).

        Returns
        -------
        out : np.ndarray
            Concatenated data per site provided in `sites`.

        """
        if PyList_CheckExact(sites) | PyTuple_CheckExact(sites):
            out = self._iterate_and_append(sites, False)
            out = np.PyArray_Concatenate(out, 0)
        else:
            out = <object>PyDict_GetItem(self.data, <PyObject*>sites)

        return out

    def __len__(self):
        """Total number of surveyed sites."""
        return PyDict_Size(self.data)

    def __reduce__(self):
        return self.__class__, (self._pickle_data,)
