# MixerStream - Real-time PCM mixing with I2S output
# Mixes any number of 8/16/32-bit mono PCM sample buffers together and
# writes the result to I2S continuously from a dedicated thread.
# 8-bit samples are unsigned, 16/32-bit samples are signed little-endian
# (same convention as WAV PCM).
#
# Voices are mixed in mono internally, but the I2S peripheral itself is
# always configured STEREO, with every mixed sample duplicated into both
# channels directly by _mix_chunk (L=R): some I2S DACs (e.g. the CJC4344-type
# chip on the Fri3d Camp 2026 badge) produce no audible output at all from a
# bare mono I2S stream, matching what WAVStream.play() does based on the
# source file's own channel count.
# Mirroring WAVStream.play(), the MCLK line is also driven via PWM
# whenever the output declares an 'mck' pin -- some DACs are disabled
# entirely without it.

import _thread
import logging
import machine
import micropython
import sys
import time

logger = logging.getLogger(__name__)


class _Voice:
    def __init__(self, voice_id, data, gain, loop, on_complete, persistent):
        self.id = voice_id
        self.data = data
        self.pos = 0
        self.gain = gain  # Q8 fixed-point (256 == 100%)
        self.loop = loop
        self.on_complete = on_complete
        self.persistent = persistent  # if True, survives past finishing so it can be re-triggered
        self.active = False


class MixerStream:
    """
    Real-time PCM mixer with I2S output.

    Voices can be added/removed from any thread while the mixing loop runs
    on its own dedicated thread; access to the voice list is guarded by a lock.
    """

    def __init__(
        self,
        volume,
        i2s_pins,
        sample_rate,
        on_open=None,
        on_close=None,
    ):
        self.volume = volume
        self.i2s_pins = i2s_pins
        self.sample_rate = sample_rate
        self.on_open = on_open
        self.on_close = on_close

        self._lock = _thread.allocate_lock()
        self._voices = {}
        self._next_voice_id = 1
        self._keep_running = True
        self._is_running = False
        self._i2s = None
        self._mck_pwm = None

        # Reused every chunk instead of allocating fresh buffers on the hot path
        # (the mixing thread runs continuously, so this matters even at idle).
        # Interleaved stereo, 16-bit per channel -> 4 bytes per output sample.
        self._chunk_samples = 1024
        self._mix_buffer = bytearray(self._chunk_samples * 4)
        self._silence = bytes(self._chunk_samples * 4)

    def is_running(self):
        return self._is_running

    def stop(self):
        self._keep_running = False

    def set_volume(self, volume):
        self.volume = volume

    # ------------------------------------------------------------------
    #  Voice management (safe to call from any thread)
    # ------------------------------------------------------------------
    def play_voice(self, samples, bits_per_sample=16, volume=100, loop=False, on_complete=None):
        """One-shot voice: starts playing immediately, and is discarded automatically
        once it finishes (or is stopped). Use add_voice()/start_voice() instead for
        a sample that will be (re-)triggered repeatedly, e.g. a sequencer step."""
        return self._new_voice(samples, bits_per_sample, volume, loop, on_complete, persistent=False, autostart=True)

    def add_voice(self, samples, bits_per_sample=16, volume=100, loop=False, on_complete=None, autostart=False):
        """
        Register a reusable voice up front (e.g. one of a fixed set of sequencer
        sample slots). The PCM conversion happens once, here. The returned voice_id
        can be passed to start_voice() any number of times - including once per
        several different voice_ids back-to-back for simultaneous triggers - to
        (re-)play it from the beginning. Unlike play_voice(), a created voice is
        never discarded when it finishes; call remove_voice() to free it.
        """
        return self._new_voice(samples, bits_per_sample, volume, loop, on_complete, persistent=True, autostart=autostart)

    def _new_voice(self, samples, bits_per_sample, volume, loop, on_complete, persistent, autostart):
        data = self._to_pcm16(samples, bits_per_sample)
        gain = self._volume_to_gain(volume)
        voice = _Voice(0, data, gain, loop, on_complete, persistent)
        voice.active = autostart
        with self._lock:
            voice_id = self._next_voice_id
            self._next_voice_id += 1
            voice.id = voice_id
            self._voices[voice_id] = voice
        return voice_id

    def start_voice(self, voice_id, volume=None):
        """(Re-)start a voice from the beginning. Safe to call while it is already playing."""
        with self._lock:
            voice = self._voices.get(voice_id)
            if voice is None:
                return False
            if volume is not None:
                voice.gain = self._volume_to_gain(volume)
            voice.pos = 0
            voice.active = True
        return True

    def start_voices(self, voice_ids):
        """(Re-)start several voices from the beginning together, e.g. every
        track that fires on the same sequencer step. One lock acquisition for
        the whole batch instead of one per voice, and they land in the same
        mixed chunk instead of possibly being split across two if the mixing
        thread's next chunk starts partway through a series of individual
        start_voice() calls."""
        with self._lock:
            for voice_id in voice_ids:
                voice = self._voices.get(voice_id)
                if voice is not None:
                    voice.pos = 0
                    voice.active = True

    def stop_voice(self, voice_id):
        """Stop a voice. A persistent voice (from add_voice()) is kept around
        inactive so start_voice() can replay it later; a one-shot voice (from
        play_voice()) is discarded immediately, same as when it finishes naturally."""
        with self._lock:
            voice = self._voices.get(voice_id)
            if voice is None:
                return
            voice.active = False
            if not voice.persistent:
                self._voices.pop(voice_id, None)

    def remove_voice(self, voice_id):
        """Fully unregister a voice, freeing its sample data."""
        with self._lock:
            self._voices.pop(voice_id, None)

    def is_voice_active(self, voice_id):
        with self._lock:
            voice = self._voices.get(voice_id)
            return voice is not None and voice.active

    def set_voice_volume(self, voice_id, volume):
        gain = self._volume_to_gain(volume)
        with self._lock:
            voice = self._voices.get(voice_id)
            if voice is not None:
                voice.gain = gain

    def clear_voices(self):
        with self._lock:
            self._voices = {}

    def active_voice_count(self):
        with self._lock:
            return sum(1 for v in self._voices.values() if v.active)

    @staticmethod
    def _volume_to_gain(volume):
        return max(0, min(256, int(volume * 256 // 100)))

    @staticmethod
    def _to_pcm16(samples, bits_per_sample):
        if bits_per_sample == 16:
            return samples if isinstance(samples, (bytes, bytearray)) else bytes(samples)
        from mpos.audio.stream_wav import WAVStream
        if bits_per_sample == 8:
            return WAVStream._convert_8_to_16(bytearray(samples))
        if bits_per_sample == 32:
            return WAVStream._convert_32_to_16(bytearray(samples))
        raise ValueError("Unsupported bits_per_sample: %s (must be 8, 16 or 32)" % bits_per_sample)

    # ------------------------------------------------------------------
    #  Mixing
    # ------------------------------------------------------------------
    @staticmethod
    @micropython.native
    def _mix_chunk(voices, out, num_samples, master_gain):
        """Mix currently-active voices directly into `out`, an interleaved-stereo
        16-bit PCM buffer (L=R, since the mixing itself is mono) that the caller
        has already zeroed. Advances each voice's position and clears .active on
        any non-looping voice that reaches its end.

        Returns finished_voices - voices that just transitioned to inactive.
        """
        finished = []

        for voice in voices:
            if not voice.active:
                continue

            data = voice.data
            data_len = len(data)
            pos = voice.pos
            gain = (voice.gain * master_gain) >> 8
            if gain <= 0:
                continue

            out_idx = 0
            for _ in range(num_samples):
                if pos >= data_len:
                    if voice.loop:
                        pos = 0
                    else:
                        break

                sample = data[pos] | (data[pos + 1] << 8)
                if sample >= 32768:
                    sample -= 65536
                sample = (sample * gain) >> 8

                existing = out[out_idx] | (out[out_idx + 1] << 8)
                if existing >= 32768:
                    existing -= 65536

                mixed = existing + sample
                if mixed > 32767:
                    mixed = 32767
                elif mixed < -32768:
                    mixed = -32768
                if mixed < 0:
                    mixed += 65536

                lo = mixed & 0xFF
                hi = (mixed >> 8) & 0xFF
                out[out_idx] = lo
                out[out_idx + 1] = hi
                out[out_idx + 2] = lo
                out[out_idx + 3] = hi

                pos += 2
                out_idx += 4

            voice.pos = pos
            if pos >= data_len and not voice.loop:
                voice.active = False
                finished.append(voice)

        return finished

    def _mix_next_chunk(self):
        with self._lock:
            voices = list(self._voices.values())

        if not any(v.active for v in voices):
            # Common case for a sequencer between triggers: nothing to mix,
            # hand back the cached all-zero chunk instead of doing any work.
            return self._silence

        out = self._mix_buffer
        out[:] = self._silence  # cheap in-place memcpy reset, no allocation

        finished = self._mix_chunk(voices, out, self._chunk_samples, self.volume * 256 // 100)

        if finished:
            with self._lock:
                for voice in finished:
                    if not voice.persistent:
                        self._voices.pop(voice.id, None)
            for voice in finished:
                if voice.on_complete:
                    try:
                        voice.on_complete()
                    except Exception as e:
                        logger.error("Voice on_complete failed: %s", e)

        return out

    # ------------------------------------------------------------------
    #  Main loop (runs in a dedicated thread)
    # ------------------------------------------------------------------
    def run(self):
        self._is_running = True
        try:
            if sys.platform != "esp32":
                self._run_desktop()
            else:
                self._run_i2s()
        finally:
            self._is_running = False
            if self.on_close:
                try:
                    self.on_close()
                except Exception as e:
                    logger.error("on_close failed: %s", e)

    def _run_i2s(self):
        ibuf = 8192 * 2

        if self.on_open:
            try:
                self.on_open()
            except Exception as e:
                logger.error("on_open failed: %s", e)

        if "mck" in self.i2s_pins:
            from machine import PWM
            from mpos.audio.stream_wav import WAVStream
            mck_pin = machine.Pin(self.i2s_pins["mck"], machine.Pin.OUT)
            try:
                self._mck_pwm = PWM(mck_pin)
                freq, duty = WAVStream._get_freq_duty(self.sample_rate)
                self._mck_pwm.freq(freq)
                self._mck_pwm.duty_u16(duty)
            except Exception as e:
                logger.error("MCLK PWM init failed: %s", e)

        try:
            if self.i2s_pins.get("sck"):
                self._i2s = machine.I2S(
                    0,
                    sck=machine.Pin(self.i2s_pins['sck'], machine.Pin.OUT),
                    ws=machine.Pin(self.i2s_pins['ws'], machine.Pin.OUT),
                    sd=machine.Pin(self.i2s_pins['sd'], machine.Pin.OUT),
                    mode=machine.I2S.TX,
                    bits=16,
                    format=machine.I2S.STEREO,
                    rate=self.sample_rate,
                    ibuf=ibuf,
                )
            else:
                self._i2s = machine.I2S(
                    0,
                    ws=machine.Pin(self.i2s_pins['ws'], machine.Pin.OUT),
                    sd=machine.Pin(self.i2s_pins['sd'], machine.Pin.OUT),
                    mode=machine.I2S.TX,
                    bits=16,
                    format=machine.I2S.STEREO,
                    rate=self.sample_rate,
                    ibuf=ibuf,
                )
        except Exception as e:
            logger.error("I2S init failed: %s", e)
            return

        try:
            while self._keep_running:
                chunk = self._mix_next_chunk()
                remaining = memoryview(chunk)
                while remaining and self._keep_running:
                    written = self._i2s.write(remaining)
                    if written is None:
                        written = len(remaining)
                    if written <= 0:
                        time.sleep_ms(1)
                        continue
                    remaining = remaining[written:]
        finally:
            if self._i2s:
                if __debug__: logger.debug("Mixer stopped, doing i2s deinit")
                self._i2s.deinit()
                self._i2s = None
            if self._mck_pwm:
                try:
                    self._mck_pwm.deinit()
                except Exception:
                    pass
                self._mck_pwm = None

    def _run_desktop(self):
        # No real audio output on desktop; just run the mixing loop at
        # roughly real-time pace so voice progress/completion behaves the same.
        interval = self._chunk_samples / self.sample_rate
        while self._keep_running:
            self._mix_next_chunk()
            time.sleep(interval)
