from __future__ import print_function
import midi
from midi2note import *


def main():
  path = 'mary.mid'

  pattern = midi.read_midifile(path)
  # print(pattern)
  
  note_states = midiToNoteStateMatrix(path)
  path2 = "mary2.mid"
  noteStateMatrixToMidi(note_states, "mary2")
  pattern = midi.read_midifile(path2)
  print(pattern)



if __name__ == '__main__':
  main()

