cimport cython
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[float] apply_avg_bert(
    np.ndarray[float] col_asid,
    np.ndarray[float] col_stream
  ):

  # retrieve record count
  cdef int nrecords = len(col_asid)

  # Create empty array to return
  cdef np.ndarray[float] res = np.empty(nrecords)

  # Fill the empty array
  for i in range(nrecords):
    res[i] = avg_bert_us(col_asid[i], col_stream[i])

  return res