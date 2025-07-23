import numpy as np
import soundfile as sf
import torchaudio
import torch
from scipy.signal import get_window
from scipy.stats import mode, pearsonr
from scipy.optimize import least_squares
from demucs.pretrained import get_model
from demucs.apply import apply_model

# ----- Functions -----
def compute_lag(x, y, sr=44100, max_tau_sec=20):
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

def residuals(alpha, y1, y2):
    return (alpha * y1) - y2

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

def framewise_lag(y1, y2, sr=44100, frame_dur=1.0, hop_dur=0.5, rms_threshold=1e-6):
    assert len(y1) == len(y2)

    frame_size = int(frame_dur * sr)
    hop_size = int(hop_dur * sr)
    y1_frames = frame_signal(y1, frame_size, hop_size)
    y2_frames = frame_signal(y2, frame_size, hop_size)
    n_frames = y1_frames.shape[0]
    
    lags = []
    
    for i in range(n_frames):
        f1 = y1_frames[i, :]
        f2 = y2_frames[i, :]
    
        if rms(f1) > rms_threshold and rms(f2) > rms_threshold:
            frame_lag = compute_lag(f1, f2, sr, max_tau_sec=0.25)
            lags.append(frame_lag)

    return mode(lags, keepdims=True)[0][0]

# ----- main -----

# Load both full-version audio (Live, Recorded)
Song = "Twice" 
Live_input = f"{Song}/Live.wav"
Rec_input = f"{Song}/AR.wav"

waveform_1, sr_1 = torchaudio.load(Live_input)
waveform_2, sr_2 = torchaudio.load(Rec_input)

target_sr = 44100

if sr_1 != target_sr:
    resampler_1 = torchaudio.transforms.Resample(orig_freq=sr_1, new_freq=target_sr)
    waveform_1 = resampler_1(waveform_1)
    sr_1 = target_sr 

if sr_2 != target_sr:
    resampler_2 = torchaudio.transforms.Resample(orig_freq=sr_2, new_freq=target_sr)
    waveform_2 = resampler_2(waveform_2)
    sr_2 = target_sr
    
def run_demucs(waveform):
     
    model = get_model(name="htdemucs").cuda()
    model.eval()

    # 1. Apply separation
    waveform = waveform.unsqueeze(0)
    sources = apply_model(model, waveform, split=True, overlap=0.25, progress=True)[0]

    # 2. Map sources to names
    sources_dict = dict(zip(model.sources, sources))

    # 3. Combine drums, bass, and other → accompaniment
    rest = sources_dict["drums"] + sources_dict["bass"] + sources_dict["other"]
    vocals = sources_dict["vocals"]
    
    return vocals, rest

# Separate Vocals and Accompaniments
Live_vox, Live_inst = run_demucs(waveform_1)
Rec_vox, Rec_inst = run_demucs(waveform_2)

# Convert from Tensor to Numpy
Live_vox = Live_vox.detach().cpu().numpy().T
Live_inst = Live_inst.detach().cpu().numpy().T
Rec_vox = Rec_vox.detach().cpu().numpy().T
Rec_inst = Rec_inst.detach().cpu().numpy().T

# Convert to Mono
vox_live = np.mean(Live_vox, axis=1)
inst_live = np.mean(Live_inst, axis=1)
vox_rec = np.mean(Rec_vox, axis=1)
inst_rec = np.mean(Rec_inst, axis=1)

# Compute lag between instruments - apply
lag_inst = compute_lag(inst_live, inst_rec, sr=44100, max_tau_sec=20)
vox_live_instlag, vox_rec_instlag = apply_lag(vox_live, vox_rec, lag=lag_inst)

# Amplitude Matching
alpha = compute_scale_factor(vox_rec_instlag, vox_live_instlag)
vox_rec_instlag_amp_match = vox_rec_instlag * alpha

# Compute Framewise - lag
lag_fw = framewise_lag(vox_live_instlag, vox_rec_instlag_amp_match)
vox_live_instlag_voxlagfw, vox_rec_instlag_amp_match_voxlagfw = apply_lag(vox_live_instlag, vox_rec_instlag_amp_match, lag=lag_fw)
output_signal = vox_live_instlag_voxlagfw - vox_rec_instlag_amp_match_voxlagfw

# Save .wav
sf.write(f"{Song}_Separted.wav", output_signal, samplerate=44100)