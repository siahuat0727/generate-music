import midi

ON  = 1
OFF = 0
EOT = -1

def track_append(track, evt_type, tick=1, pitch=0):
  if evt_type == ON:
    evt = midi.NoteOnEvent(tick=tick, velocity=40, pitch=pitch)
  elif evt_type == OFF:
    evt = midi.NoteOffEvent(tick=tick, velocity=40, pitch=pitch)
  elif evt_type == EOT:
    evt = midi.EndOfTrackEvent(tick=tick)
  else:
    print('Unexpected event type!')
  track.append(evt)

def main():
  pattern = midi.Pattern()
  track = midi.Track()
  pattern.append(track)

  pitchs = [-5,-5,-6,1,2,3,6,5,3,2,1,1,1,2,1,-7,1,4,-6,1,3,3,2,2,2,3,2,0,2,5,-7,2,3,2,1,1,1,2,1,-7,1,4,-6,1,-6,-6,-5,-5,-5,-5,-6,1,2,3,6,5,3,2,1,1,1,2,1,-7,1,4,-6,1,3,3,2,2,2,3,2,0,2,5,-7,2,3,2,1,1,1,2,1,-7,1,4,-6,1,2,2,1]
  convert = {0:2, 1:1, 2:3, 3:5, 4:6, 5:8, 6:10, -5:-4, -6:-2, -7:0}
  pitchs = [convert[x] for x in pitchs]
  pitchs = [p+60 for p in pitchs]
  tempos = [2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2,6,2,2,2,2,2,4,2,4,2,2,2]
  # tempos = [2*x for x in tempos]
  print(len(pitchs))
  print(len(tempos))
  # repeats = [4, 3, 6, 3]
  print('resolution = ', pattern.resolution)
  idx = 0

  hmn = [0]
  pitchs = [tuple(p+i for i in hmn) for p in pitchs]
  print(pitchs)
  for _ in range(100):
    for ps, t in zip(pitchs, tempos):
      for p in ps:
        track_append(track, ON, 0, p)
      for idx, p in enumerate(ps):
        track_append(track, OFF, 55*t if idx==0 else 0, p)

  track_append(track, EOT, tick=1)
  midi.write_midifile("shoushou.mid", pattern)

if __name__ == '__main__':
  main()

  

