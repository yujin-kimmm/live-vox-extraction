import numpy as np
from scipy.stats import mode, pearsonr
from scipy.optimize import least_squares
from utils import frame_signal, rms

def compute_scale_factor(y1, y2, sr=44100, frame_dur=1.0, hop_dur=0.5, rms_threshold=1e-6, show_plot=True):
    # y1 = rec, y2 = live for this
    assert len(y1) == len(y2)

    frame_size = int(frame_dur * sr)
    hop_size = int(hop_dur * sr)
    y1_frames = frame_signal(y1, frame_size, hop_size)
    y2_frames = frame_signal(y2, frame_size, hop_size)
    n_frames = y1_frames.shape[0]
    
    corrs = []

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
    
    lsq_sol = least_squares(residuals, alpha_0,
                            args=(y1_frames[max_corr_frame, :],
                                  y2_frames[max_corr_frame, :]))
    return lsq_sol.x

def residuals(alpha, y1, y2):
    return (alpha * y1) - y2