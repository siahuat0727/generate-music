from __future__ import print_function
import numpy as np
from midi_statematrix import span
import random

def state2input(state):
  """list to numpy"""
  return np.asarray(state).reshape((-1))

def print_state(state):
  printed = False
  for idx, s in enumerate(state):
    if s == [0, 0]:
      continue
    print('%s [%d, %d] ' % (num2key(idx), s[0], s[1]), end='')
    printed = True
  if printed:
    print()

def num2key(num):
  return '%c%d' % (chr(ord('A') + num%12), num//12)


def input2state(input_):
  "numpy to list"
  input_ = input_.reshape(-1, 2).tolist()
  input_ = [[1 if x > 0.5 else 0 for x in i] for i in input_]
  return input_

def input2string(input_):
  """Debug use"""
  s = "in2str:"
  for i, x in enumerate(input_):
    if x == 1:
      s += chr(i%24 + ord('A'))
      s += str(i//24)
      s += str(i//2)
  return s

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, 2*span], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

