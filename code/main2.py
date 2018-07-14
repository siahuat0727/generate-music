from __future__ import print_function
from midi_statematrix import *
from lib import *

def main():
  ms = midiToNoteStateMatrix('../tiger.mid')
  noteStateMatrixToMidi(ms, name="tiger2")
  # ms = [[[0,0] for _ in range(span)] for _ in range(100)]
  idx = 0
  for i in range(100):
    ms[i][idx][0] = 1
    idx = (idx+5) % 78
  for idx, m in enumerate(ms):
    print(idx)
    print_state(m)

if __name__ == '__main__':
  main()



