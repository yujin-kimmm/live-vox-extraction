import numpy as np
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model


def separate(file, gpu=None):
    """
    Run Demucs source separation model on WAV files to isolate stems.
    
    Parameters
    ----------
    file :  str
            Path to the mixture WAV file to separate.
    
    gpu : torch.device
          GPU device to use for processing. If None, CPU will be used.
          
    Returns
    -------
    vocals : np.ndarray
            Isolated vocal signal, converted to mono.
    
    accompaniments : np.ndarray
                    Isolated and combined signal (drums, bass, other), converted to mono.
                    
    sr : int
         Sample rate
    """
    # only process WAV files
    if file.endswith(".wav"):
        waveform, sr = torchaudio.load(file)
    
    if gpu: 
        model = get_model(name="htdemucs").cuda()
    else:
        model = get_model(name="htdemucs").cpu()

    model.eval()

    
    # Apply separation
    waveform = waveform.unsqueeze(0)
    sources = apply_model(model, waveform, split=True, overlap=0.25, progress=True)[0]

    # Map sources to names
    sources_dict = dict(zip(model.sources, sources))

    # Combine drums, bass, and other → accompaniment
    accompaniments = sources_dict["drums"] + sources_dict["bass"] + sources_dict["other"]
    vocals = sources_dict["vocals"]
    
    # Convert from Tensor to Numpy
    vocals = vocals.detach().cpu().numpy().T
    accompaniments = accompaniments.detach().cpu().numpy().T
    
    # Convert to mono
    if len(vocals.shape) != 1:
        vocals = np.mean(vocals, axis=1)
    
    if len(accompaniments.shape) != 1:
        accompaniments = np.mean(accompaniments, axis=1)
    
    return vocals, accompaniments, sr
