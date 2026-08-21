# Quick manual test: play multiple PCM voices at the same time through
# AudioManager.mixer(). Assumes the board's default i2s audio output is
# already registered (done automatically at boot by mpos.board.*).
#
# Run on-device via: mpremote run test_mixer.py

import math
import time

from mpos import AudioManager

SAMPLE_RATE = 16000


def make_tone(frequency, duration_s, sample_rate=SAMPLE_RATE, amplitude=8000):
    """Generate a mono 16-bit signed PCM sine wave buffer."""
    num_samples = int(sample_rate * duration_s)
    buf = bytearray(num_samples * 2)
    for i in range(num_samples):
        sample = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
        buf[i * 2] = sample & 0xFF
        buf[i * 2 + 1] = (sample >> 8) & 0xFF
    return buf


mixer = AudioManager.mixer(sample_rate=SAMPLE_RATE)
mixer.start()

# --- Part 1: simultaneous one-shot voices -----------------------------------
# A C major chord: three notes played at the same time as three separate voices.
# Each voice is turned down so the sum doesn't clip.
c4 = make_tone(261.63, 2.0)
e4 = make_tone(329.63, 2.0)
g4 = make_tone(392.00, 2.0)

mixer.play(c4, bits_per_sample=16, volume=40)
mixer.play(e4, bits_per_sample=16, volume=40)
mixer.play(g4, bits_per_sample=16, volume=40)

print("Playing C major chord (3 voices)...")
time.sleep(2.5)

# A short beep looped on top while other voices could still be playing;
# stop_voice() lets you cut a specific voice without touching the rest.
beep = make_tone(880, 0.1)
beep_voice = mixer.play(beep, bits_per_sample=16, volume=60, loop=True)
print("Looping beep voice for 1s...")
time.sleep(1)
mixer.stop_voice(beep_voice)


# --- Part 2: sequencer-style, pre-loaded, re-triggerable voices -------------
# add_voice() registers and PCM-converts a sample once, up front - ideal for
# a fixed bank of samples (e.g. drum hits) that get triggered many times, and
# sometimes several at once, over the course of a sequence.
# start_voice() just resets that voice's playback position and is cheap enough
# to call on every beat; calling it for several voice_ids back-to-back triggers
# them simultaneously. Retriggering a voice that's still playing restarts it
# from the beginning (a real sample file would replace make_tone() here).
NUM_TRACKS = 8
voice_ids = [
    mixer.add_voice(make_tone(220 * (track + 1), 0.05), bits_per_sample=16, volume=70)
    for track in range(NUM_TRACKS)
]

# 8 steps x 8 tracks; 1 = trigger that track's sample on that step.
pattern = [
    [1, 0, 0, 0, 1, 0, 0, 0],  # step 0: tracks 0 and 4 together
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],  # step 2: tracks 2 and 5 together
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 0],  # step 4: tracks 0, 3 and 4 together
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1],
]

step_duration_s = 0.15
print("Playing an 8-step / 8-track sequence, looped 50 times...")
for loop_iteration in range(50):
    for step in pattern:
        # start_voices() triggers every hit on this step in one call, so they
        # land in the same mixed chunk instead of possibly being split across
        # two if the mixing thread's next chunk starts mid-way through a
        # series of individual start_voice() calls.
        mixer.start_voices([voice_ids[track] for track, hit in enumerate(step) if hit])
        time.sleep(step_duration_s)

for voice_id in voice_ids:
    mixer.remove_voice(voice_id)

mixer.stop()
print("Done.")
