import soundfile as sf
import torch
import os
import argparse
from separate import separate
from shift import compute_lag, apply_lag, framewise_lag
from scale import compute_scale_factor
from utils import check_sample_rate


SAMPLE_RATE = 44100


def main():
    """
    Parse command-line arguments and extract the live singing vocal from live performance.
    """
    # parse arguments
    parser = argparse.ArgumentParser(description='Setting input and output directory to run Live Vocal Separation.')
    
    parser.add_argument("--live", help='Path to the Live audio file')
    parser.add_argument("--recorded", help='Path to the Recorded-audio audio file')
    parser.add_argument("-o", "--output_dir", default='./Separated',
                        help='Path to the output directory to save the separated vocals.(default: ./Separated)')
    
    args = parser.parse_args()
    
    # Separate the loaded audio files
    print("Separating Live audio start.")
    Live_vox, Live_inst, sr_1 = separate(args.live, DEVICE)
    print("Separating Live audio done.")
    
    print("Separating Recorded audio start.")
    Rec_vox, Rec_inst, sr_2 = separate(args.recorded, DEVICE)
    print("Separating Recorded audio done.")
    
    print("Separating vocals...")
    
    
    # Check if sr is 44100
    # Resample to 44100 if sr is not 44100
    vox_live, inst_live, sr_1 = check_sample_rate(Live_vox, Live_inst, sr_1, SAMPLE_RATE)   
    vox_rec, inst_rec, sr_2 = check_sample_rate(Rec_vox, Rec_inst, sr_2, SAMPLE_RATE)
    
    # Compute lag between instruments - apply
    lag_inst = compute_lag(inst_live, inst_rec, sr=SAMPLE_RATE, max_tau_sec=20)
    
    # applying lag
    vox_live_instlag, vox_rec_instlag = apply_lag(vox_live, vox_rec, lag=lag_inst)

    # Compute scale factor
    alpha = compute_scale_factor(vox_rec_instlag, vox_live_instlag)
    
    # Multiplying alpha to recorded vox
    vox_rec_instlag_amp_match = vox_rec_instlag * alpha
        
    # Compute Framewise - lag
    lag_fw = framewise_lag(vox_live_instlag, vox_rec_instlag_amp_match)
    
    # Applying lag to recorded vox
    vox_live_instlag_voxlagfw, vox_rec_instlag_amp_match_voxlagfw = apply_lag(vox_live_instlag, vox_rec_instlag_amp_match, lag=lag_fw)
    
    # Subtraction
    output_signal = vox_live_instlag_voxlagfw - vox_rec_instlag_amp_match_voxlagfw
    print("Vocals separated succesfully.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "Extracted_live_vocal.wav")
    sf.write(output_path, output_signal, samplerate=SAMPLE_RATE)
    
    print(f"Extracted Live vocals succesfully saved in {output_dir}.")
    
    
if __name__ == '__main__':

    # gpu check
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("GPU detected: using CUDA.")
    
    else:
        DEVICE = None
        print("GPU not found. Defaulting to CPU.")

    main()
    