import numpy as np
from batch_generator import BatchGenerator

class MixedGenarator(object):
  def __init__(self, pieces, batch_size, num_unrolling):
    self._batch_generator = list()
    batch_size_each = batch_size // len(pieces) # TODO exactly fix batch_size
    if batch_size_each == 0:
      batch_size_each = 1
    for p in pieces:
      self._batch_generator.append(BatchGenerator(p, batch_size_each, num_unrolling))

  def next(self):
    batches = self._batch_generator[0].next()
    for batch_gen in self._batch_generator[1:]:
      for i, t in enumerate(zip(batches, batch_gen.next())):
        batches[i] = np.concatenate((t[0], t[1]))
    return batches


    



