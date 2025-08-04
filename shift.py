import numpy as np
from scipy.stats import mode
from utils import frame_signal, rms


def compute_lag(x, y, sr=44100, max_tau_sec=20):
    """
    Compute lag between two signals using 
    Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    
    Parameters
    ----------
    x : np.ndarray
        Audio signal 1.
                
    y : np.ndarray
        Audio signal 2.
                
    sr : int, optional (default 44100 Hz)
         Sampling Rate.
        
    max_tau_sec : int (default 20 sec)
                  Maximum tau in seconds.
    
    Return
    ------
    lag : int
        Computed lag between x and y.
    """
    # Ensure inputs are 1D numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)

    # Length for zero-padding
    n = x.shape[0] + y.shape[0] - 1

    # FFT of both signals
    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)

    # GCC-PHAT: cross-power spectrum normalized by magnitude
    R = X * np.conj(Y)
    R /= np.abs(R) + 1e-15  # avoid division by zero

    # Inverse FFT gives cross-correlation in time domain
    cc = np.fft.irfft(R, n=n)

    # Shift zero-lag to center
    cc = np.concatenate((cc[-(len(y) - 1):], cc[:len(x)]))

    # Apply lag limit (in samples)
    max_lag = int(max_tau_sec * sr)
    center = len(cc) // 2
    lower_bound = center - max_lag
    upper_bound = center + max_lag

    # Mask everything outside [-max_lag, +max_lag]
    mask = np.zeros_like(cc)
    mask[lower_bound:upper_bound] = 1
    cc *= mask
    
    # Find lag with maximum correlation
    lag = np.argmax(cc) - (len(y) - 1)
    corr = cc[lag]

    return lag


def apply_lag(x, y, lag):
    """
    Applying lag by padding to align two signals. 
    
    Parameters
    ----------
    x : np.ndarray
        Audio signal 1.
        
    y : np.ndarray
        Audio signal 2.
        
    lag : int
          Computed lag between two signal from Compute_lag function.
    
    Returns
    -------
    x : np.ndarray
        aligned signal 1 after applying lag.
        
    y : np.ndarray
        aligned signal 2 after applying lag.
    """
    
    if lag > 0:
        y = np.pad(y, (lag, 0), mode='constant') 
    elif lag < 0:
        x = np.pad(x, (abs(lag), 0), mode='constant')  

    # Pad the end to equalize lengths
    len1, len2 = len(x), len(y)
    if len1 < len2:
        x = np.pad(x, (0, len2 - len1), mode='constant')
    elif len2 < len1:
        y = np.pad(y, (0, len1 - len2), mode='constant')

    return x, y


def framewise_lag(y1, y2, sr=44100, frame_dur=1.0, hop_dur=0.5, rms_threshold=1e-6):
    """
    Compute lag between two signals by analizing frame-by-frame lag using cross correlation. 
    
    Parameters
    ----------
    y1 : np.ndarray
        audio signal 1.
        
    y2 : np.ndarray
        audio signal 2.
        
    sr : int, optional (default 44100 Hz)
        Sampling rate.
        
    frame_dur : float, optional (default 1.0 sec)
                Duration of each frame in seconds.
        
    hop_dur : float, optional (default 0.5 sec)
                Step size (hop) between frames in seconds.
        
    rms_threshold : float, optional (default 1e-6)
                    RMS threshold to avoid computing the lag in silent frames.
        
    Return
    ------
    int
        Computed frame-by-frame lag between y1 and y2. 
    
    """
    
    assert len(y1) == len(y2)

    frame_size = int(frame_dur * sr)
    hop_size = int(hop_dur * sr)
    y1_frames = frame_signal(y1, frame_size, hop_size)
    y2_frames = frame_signal(y2, frame_size, hop_size)
    n_frames = y1_frames.shape[0]
    
    lags = []
    
    # Iterate through each frames
    for i in range(n_frames):
        f1 = y1_frames[i, :]
        f2 = y2_frames[i, :]

        # Avoid the frame that has no signal (silence)
        if rms(f1) > rms_threshold and rms(f2) > rms_threshold:
            frame_lag = compute_lag(f1, f2, sr, max_tau_sec=0.25)
            lags.append(frame_lag)

    return mode(lags, keepdims=True)[0][0]
