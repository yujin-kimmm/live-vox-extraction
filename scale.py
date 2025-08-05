import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import least_squares
from utils import frame_signal, rms


def residuals(alpha, y1, y2): # y1 = rec, y2 = live for this function
    """
    Compute the residuals between scale factor-multiplied signal and target signal.
    This code will be used to compute least sqaure in compute_scale_factor function.
    
    Parameters
    ----------
    alpha : float
            Scale factor to be optimized
    
    y1 : np.ndarray
        Input signal to be scaled.
        
    y2 : np.ndarray
        Target signal to match after scaling
    
    Return
    ------
    np.ndarray
    
    """
    
    return (alpha * y1) - y2


def compute_scale_factor(y1, y2, sr=44100, frame_dur=1.0, hop_dur=0.5, rms_threshold=1e-6):  # y1 = rec, y2 = live for this function
    """
    Compute the optimal scale factor to minimize the differences of amplitude 
    between two signals using frame-wise correlation.
    
    Parameters
    ----------
    y1 : np.ndarray
        Signal that needs to be multiplied by alpha. 
        
    y2 : np.ndarray
        Target signal.
        
    sr : int, optional (default 44100 Hz)
        Sampling ratee.
    
    frame_dur : float, optional (default 1.0 sec)
                Duration of each frame in seconds.
        
    hop_dur : float, optional (default 0.5 sec)
                Step size (hop) between frames in seconds.
        
    rms_threshold : float, optional (default 1e-6)
                    RMS threshold to avoid computing the lag in silence.
                    
    Return
    ------
    float
        Scale factor (alpha) that minimizes the differences 
        of amplitude between two signals.
    """
    
    assert len(y1) == len(y2)

    frame_size = int(frame_dur * sr)
    hop_size = int(hop_dur * sr)
    y1_frames = frame_signal(y1, frame_size, hop_size)
    y2_frames = frame_signal(y2, frame_size, hop_size)
    n_frames = y1_frames.shape[0]
    
    corrs = []
    
    # Find the frame that has highest cross correlation
    for i in range(n_frames):
        f1 = y1_frames[i, :]
        f2 = y2_frames[i, :]
    
        if rms(f1) < rms_threshold or rms(f2) < rms_threshold:
            corrs.append(0)
        else:
            r, _ = pearsonr(f1, f2)
            corrs.append(r)

    max_corr_frame = np.argmax(corrs)
    # print(f"Frame with max correlation: {max_corr_frame}")
    # print(f"Max correlation: {corrs[max_corr_frame]}")

    alpha_0 = 0.5
    
    # Find scale factor (alpha) by computing least square
    lsq_sol = least_squares(residuals, alpha_0,
                            args=(y1_frames[max_corr_frame, :],
                                  y2_frames[max_corr_frame, :]))
    return lsq_sol.x
