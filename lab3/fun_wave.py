import wave
import struct
import numpy as np


def save_wave(filename, signal, sample_rate=44100):
    with wave.open(filename, 'w') as wav_file:
        n_channels = 1
        sampwidth = 2
        n_frames = len(signal)
        comp_type = "NONE"
        comp_name = "not compressed"
        wav_file.setparams((n_channels, sampwidth, sample_rate, n_frames, comp_type, comp_name))
        max_amplitude = max(abs(signal))
        signal = (signal / max_amplitude) * 32767
        for s in signal:
            wav_file.writeframes(struct.pack('<h', int(s)))


def read_wave(filename):
    with wave.open(filename, 'r') as wav_file:
        n_channels, sampwidth, sample_rate, n_frames, comp_type, comp_name = wav_file.getparams()
        frames = wav_file.readframes(n_frames)
        if sampwidth == 2:              signal = np.array(struct.unpack('<' + 'h' * n_frames, frames), dtype=np.int16)
        else:
            raise ValueError("Поддерживаются только 16-битные WAV-файлы")
        signal = signal / 32767.0
    return signal, sample_rate