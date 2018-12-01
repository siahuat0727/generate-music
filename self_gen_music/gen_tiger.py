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

  pitchs = [1, 3, 5, 1, 5, 6, 8, 8, 10, 8, 6, 5, 1, 3, -4, 1]
  pitchs = [p+60 for p in pitchs]
  tempos = [2, 2, 2, 2, 2, 2, 4, 1, 1, 1, 1, 2, 2, 2, 2, 4]
  repeats = [4, 3, 6, 3]
  print('resolution = ', pattern.resolution)
  idx = 0
  tmp_pitchs = pitchs
  tmp_tempos = tempos
  pitchs, tempos = (list() for _ in range(2))
  for r in repeats:
    pitchs += tmp_pitchs[idx:idx+r]*2
    tempos += tmp_tempos[idx:idx+r]*2
    idx += r

  hmn = [0]
  pitchs = [tuple(p+i for i in hmn) for p in pitchs]
  print(pitchs)
  for _ in range(100):
    for ps, t in zip(pitchs, tempos):
      for p in ps:
        track_append(track, ON, 0, p)
      for idx, p in enumerate(ps):
        track_append(track, OFF, 55*t if idx==0 else 0, p)

  # hmn = [0]
  # pitchs = [tuple(p+i for i in hmn) for p in pitchs]
  # for _ in range(100):
  #   for ps, t in zip(pitchs, tempos):
  #     for p in ps:
  #       track_append(track, ON, 0, p)
  #     for idx, p in enumerate(ps):
  #       track_append(track, OFF, 55*t if idx==0 else 0, p)


  # tempos = [2, 2, 2] 
  # for p in pitchs:
  #   track_append(track, ON, 0, p)
  # track_append(track, OFF, 500, 61)

  # for p, t in zip(pitchs, [0]+tempos):
    # track_append(track, ON, 50*t, p)
    # track_append(track, OFF, 50*t, p)
  # for ps, t in zip(zip(pitchs, pitchs_2), tempos):
  #   track_append(track, ON, 0, ps[0])
  #   track_append(track, ON, 0, ps[1])
  #   track_append(track, OFF, 50*t, ps[0])
  #   track_append(track, OFF, 0, ps[1])
  track_append(track, EOT, tick=1)
  midi.write_midifile("tiger.mid", pattern)

if __name__ == '__main__':
  main()

  

