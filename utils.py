import numpy as np
from scipy.signal import get_window, resample

def frame_signal(x, frame_size, hop_size, window_fn='hann'):
    if window_fn:
        window = get_window(window_fn, frame_size)
    else:
        window = 1
    num_frames = 1 + (len(x) - frame_size) // hop_size
    frames = np.zeros((num_frames, frame_size))

    for i in range(num_frames):
        start = i * hop_size
        frames[i, :] = x[start:start + frame_size] * window

    return frames

def rms(x):
    return np.sqrt(np.mean(x ** 2))

def check_sample_rate(x, y, sr, target_sr):
    if sr != target_sr:
        num_samples_x = int(x.shape[-1] * target_sr / sr)
        x = resample(x, num=num_samples_x, axis=-1)
        num_samples_y = int(y.shape[-1] * target_sr / sr)
        y = resample(y, num=num_samples_y, axis=-1)
        sr = target_sr
    return x, y, sr