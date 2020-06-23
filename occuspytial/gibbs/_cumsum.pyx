#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True


def split_sum(double[:] arr, long[:] sections, double[:] out):
    cdef:
        double summation
        long previous, current, size = out.shape[0]
        Py_ssize_t i, j

    with nogil:
        for i in range(size):
            previous = sections[i]
            current = sections[i + 1]
            summation = 0
            for j in range(previous, current):
                summation += arr[j]
            out[i] = summation
        