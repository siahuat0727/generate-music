import os, random
from midi_statematrix import *

import signal

batch_width = 10 # number of sequences in a batch
batch_len = 16*8 # length of each sequence
division_len = 16 # interval between possible start locations

def loadPieces(dirpath):
  """return list of statematrix"""

  pieces = list()

  for fname in os.listdir(dirpath):
    if fname[-4:] not in ('.mid','.MID'):
      continue

    name = fname[:-4]

    outMatrix = midiToNoteStateMatrix(os.path.join(dirpath, fname))
    if outMatrix == None or len(outMatrix) < batch_len:
      print('%s : too short' % (fname))
      continue

    pieces.append(outMatrix)
    print "Loaded {}".format(name)

  return pieces
