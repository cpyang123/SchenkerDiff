"""Constants for music processing"""

# Meter-related constants.
DEFAULT_QUARTERS_PER_MINUTE = 120.0
DEFAULT_STEPS_PER_BAR = 16  # 4/4 music sampled at 4 steps per quarter note.
DEFAULT_STEPS_PER_QUARTER = 4

# Default absolute quantization.
DEFAULT_STEPS_PER_SECOND = 100

# Standard pulses per quarter.
# https://en.wikipedia.org/wiki/Pulses_per_quarter_note
STANDARD_PPQ = 220

# Special melody events.
NUM_SPECIAL_MELODY_EVENTS = 2
MELODY_NOTE_OFF = -1
MELODY_NO_EVENT = -2

# Other melody-related constants.
MIN_MELODY_EVENT = -2
MAX_MELODY_EVENT = 127
MIN_MIDI_PITCH = 0  # Inclusive.
MAX_MIDI_PITCH = 127  # Inclusive.
NOTES_PER_OCTAVE = 12

# Velocity-related constants.
MIN_MIDI_VELOCITY = 1  # Inclusive.
MAX_MIDI_VELOCITY = 127  # Inclusive.

# Program-related constants.
MIN_MIDI_PROGRAM = 0
MAX_MIDI_PROGRAM = 127

# Chord symbol for "no chord".
NO_CHORD = 'N.C.'

# The indices of the pitch classes in a major scale.
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]

# NOTE_KEYS[note] = The major keys that note belongs to.
# ex. NOTE_KEYS[0] lists all the major keys that contain the note C,
# which are:
# [0, 1, 3, 5, 7, 8, 10]
# [C, C#, D#, F, G, G#, A#]
#
# 0 = C
# 1 = C#
# 2 = D
# 3 = D#
# 4 = E
# 5 = F
# 6 = F#
# 7 = G
# 8 = G#
# 9 = A
# 10 = A#
# 11 = B
#
# NOTE_KEYS can be generated using the code below, but is explicitly declared
# for readability:
# NOTE_KEYS = [[j for j in range(12) if (i - j) % 12 in MAJOR_SCALE]
#              for i in range(12)]
NOTE_KEYS = [
  [0, 1, 3, 5, 7, 8, 10],
  [1, 2, 4, 6, 8, 9, 11],
  [0, 2, 3, 5, 7, 9, 10],
  [1, 3, 4, 6, 8, 10, 11],
  [0, 2, 4, 5, 7, 9, 11],
  [0, 1, 3, 5, 6, 8, 10],
  [1, 2, 4, 6, 7, 9, 11],
  [0, 2, 3, 5, 7, 8, 10],
  [1, 3, 4, 6, 8, 9, 11],
  [0, 2, 4, 5, 7, 9, 10],
  [1, 3, 5, 6, 8, 10, 11],
  [0, 2, 4, 6, 7, 9, 11]
]
