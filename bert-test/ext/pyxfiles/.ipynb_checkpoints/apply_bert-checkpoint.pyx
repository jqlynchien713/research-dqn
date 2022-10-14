cimport cython
cimport numpy as np
import numpy as np

np.import_array()  # needed to initialize numpy-API

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef np.ndarray[double] apply_bert(
    np.ndarray[double] col_asid,
    BERT
  ):

  # retrieve record count
  cdef int nrecords = len(col_asid)

  # Create empty array to return
  cdef np.ndarray[double] res = np.empty(nrecords, dtype=float)

  # Fill the empty array
  for i in range(nrecords):
    res[i] = BERT[col_asid[i]]

  return res

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

def avg_bert_us(a, b, data, BERT):
  col_item = data[(data['asid'] == a) & (data['stream'] == b)]['item_name'].to_numpy()
  apply_bert.apply_bert(col_item, BERT).mean()