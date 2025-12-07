import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

def compute_spectral_energy(waveform, n_fft=2048):
    # waveform: (1, T)
    spec = torch.stft(waveform, n_fft=n_fft, return_complex=True, window=torch.hann_window(n_fft))
    mag = torch.abs(spec)
    # Sum energy across time -> (freq_bins,)
    energy = torch.mean(mag**2, dim=1)
    return energy

def analyze_perturbations(csv_path, output_dir="plots/perturbation"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path, sep='|')
    
    # Filter for clean samples to analyze pure watermark embedding
    # 若要看 Attack 後的殘差，則不過濾
    clean_subset = df[df['attack_type'].str.contains('clean', case=False, na=False)].head(100) 
    
    perturbation_energies = []
    snr_list = []
    
    print("Analyzing perturbations...")
    for idx, row in tqdm(clean_subset.iterrows(), total=len(clean_subset)):
        orig_path = row['orig_filepath'] # 需確保路徑正確，可能需要 prepend dataset root
        wm_path = row['audio_filepath']
        
        try:
            wav_orig, sr = torchaudio.load(orig_path)
            wav_wm, _ = torchaudio.load(wm_path)
            
            # Align lengths
            min_len = min(wav_orig.shape[1], wav_wm.shape[1])
            wav_orig = wav_orig[:, :min_len]
            wav_wm = wav_wm[:, :min_len]
            
            # Calculate Residual
            residual = wav_wm - wav_orig
            
            # 1. Calculate SNR
            signal_power = torch.mean(wav_orig**2)
            noise_power = torch.mean(residual**2)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-9))
            snr_list.append(snr.item())
            
            # 2. Spectral Footprint
            res_energy = compute_spectral_energy(residual)
            perturbation_energies.append(res_energy.numpy())
            
        except Exception as e:
            print(f"Error loading {orig_path}: {e}")
            continue

    # Visualization 1: SNR Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(snr_list, kde=True, color='skyblue')
    plt.title('Distribution of Watermark SNR')
    plt.xlabel('SNR (dB)')
    plt.savefig(f"{output_dir}/snr_histogram.png")
    plt.close()

    # Visualization 2: Spectral Footprint (Freq-wise Energy Boxplot)
    if perturbation_energies:
        energies_np = np.stack(perturbation_energies) # (N_samples, Freq_bins)
        # Normalize for visualization
        energies_log = 10 * np.log10(energies_np + 1e-9)
        
        plt.figure(figsize=(12, 6))
        # 為了簡化，我們對頻帶進行 Binning (例如每 10 個頻率 bin 取平均)
        binned_energy = energies_log.reshape(energies_log.shape[0], -1, 16).mean(axis=2)
        
        sns.boxplot(data=binned_energy, fliersize=1)
        plt.title('Watermark Spectral Footprint (Residual Energy per Band)')
        plt.xlabel('Frequency Band Index (Low -> High)')
        plt.ylabel('Log Energy (dB)')
        plt.savefig(f"{output_dir}/spectral_footprint.png")
        plt.close()
        
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    # 使用你上傳的 test_results_loose.csv
    analyze_perturbations("test_results_loose.csv")