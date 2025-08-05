import numpy as np
import librosa
from scipy.signal import get_window


def frame_signal(x, frame_size, hop_size, window_fn='hann'):
    """
    Dividing the signal into frames.
    
    Parameters
    ----------
    x : np.ndarray
        Audio signal. 
    
    frame_size : int
                 Size of each frame.
    hop_size : int
                Step size (hop size) of each frame.
                
    window_fn : str, optional (default 'hann')
                Window function for each frame.

    Return
    ------
    frames : np.ndarray
             Divided signal into frames.
    """
    # Divide the signal into frames.
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
    # Compute Root Mean Square
    return np.sqrt(np.mean(x ** 2))


def check_sample_rate(x, y, sr, target_sr):
    """
    Check if two signals are sharing same sampling rate 
    as target samplerate and resample if needed.
    
    Parameters
    ----------
    x : np.ndarray
        Audio signal 1.
        
    y : np.ndarray 
        Audio signal 2.
        
    sr : int
         Sampling rate of Original signal.
         
    target_sr : int
                Targeted sampling rate.
                
    Returns
    -------
    x : np.ndarray
        Resampled audio signal 1.
        
    y : np.ndarray
        Resampled audio signal 2.
        
    sr : int
        Targeted sampling rate.
    """
    
    if sr != target_sr:
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        
    return x, y, sr
