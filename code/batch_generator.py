from lib import *
from midi_statematrix import span
import numpy as np 

class BatchGenerator(object):
  def __init__(self, state_matrix, batch_size, num_unrolling):
    self._state_matrix = state_matrix
    self._state_len = len(state_matrix)
    self._batch_size = batch_size
    self._num_unrolling = num_unrolling
    segment = self._state_len // batch_size
    self._cursor = [offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def _next_batch(self):
    batch = np.zeros(shape=(self._batch_size, 2*span), dtype=np.float32) # TODO comfirm float ?
    for b in range(self._batch_size):
      batch[b, :] = state2input(self._state_matrix[self._cursor[b]])
      self._cursor[b] = (self._cursor[b] + 1) % self._state_len
    return batch

  def next(self):
    batches = [self._last_batch]
    for step in range(self._num_unrolling):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

      
