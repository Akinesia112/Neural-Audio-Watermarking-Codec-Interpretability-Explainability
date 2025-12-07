import os
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import julius
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import EncodecModel, AutoProcessor
from audioseal import AudioSeal

# -------- CONFIG ----------
SAMPLE_RATE = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FILES_TRAIN = 30  # PCA training
N_FILES_TEST = 30   # Attack testing
K_COMPONENTS = 60   # Cutoff for PCA attacks
LIBRISPEECH_DIR = "./wav_24k" 
OUTPUT_DIR = "./wm_attack_benchmark"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------

print(f"Running on {DEVICE}...")

# Load Models
wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(DEVICE)
encodec.eval()

# ----------------- Helpers -----------------

def get_latents(wav_np):
    """Extract continuous latents from EnCodec."""
    inputs = processor(raw_audio=wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        enc_out = encodec.encoder(inputs["input_values"])
        if hasattr(enc_out, "last_hidden_state"): z = enc_out.last_hidden_state
        else: z = enc_out[0]
    return z

def decode_latents(latents):
    """
    Decode continuous latents directly through the decoder.
    The quantizer module in HF EnCodec is not callable with bandwidth args,
    so we bypass it and feed latents directly to the decoder.
    """
    with torch.no_grad():
        # Ensure correct shape
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        if latents.dim() == 3 and latents.shape[1] != encodec.config.audio_channels:
            latents = latents

        # Direct decode (skip quantizer)
        audio_out = encodec.decoder(latents)

        # Handle HF output object
        if hasattr(audio_out, "audio_values"):
            audio = audio_out.audio_values
        else:
            audio = audio_out

    return audio.squeeze().cpu().numpy()

def detect(audio_np):
    """Run AudioSeal detector."""
    x = torch.from_numpy(audio_np).float().to(DEVICE)
    if x.ndim == 1: x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2: x = x.unsqueeze(1)
    if x.dim() == 4: x = x.squeeze(2) # Safety squeeze

    with torch.no_grad():
        prob, msg = detector.detect_watermark(x, sample_rate=SAMPLE_RATE)
    return prob

def calculate_snr(clean, degraded):
    """Calculate Signal-to-Noise Ratio (Audio Quality)."""
    # Align lengths
    min_len = min(len(clean), len(degraded))
    clean = clean[:min_len]
    degraded = degraded[:min_len]
    
    noise = clean - degraded
    power_clean = np.mean(clean**2)
    power_noise = np.mean(noise**2)
    if power_noise == 0: return 100.0
    return 10 * np.log10(power_clean / power_noise)

# ----------------- 1. TRAIN PCA -----------------
print("Training PCA on clean audio...")
files = [os.path.join(r, f) for r, _, fs in os.walk(LIBRISPEECH_DIR) for f in fs if f.endswith('.wav')]
train_files = files[:N_FILES_TRAIN]
test_files = files[N_FILES_TRAIN : N_FILES_TRAIN + N_FILES_TEST]

latent_buffer = []
for f in tqdm(train_files, desc="PCA Train"):
    wav, sr = sf.read(f)
    if sr != SAMPLE_RATE: 
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    wav = wav.astype(np.float32)
    z = get_latents(wav)
    z_flat = z.squeeze(0).permute(1, 0).cpu().numpy()
    if z_flat.shape[0] > 500: z_flat = z_flat[:500]
    latent_buffer.append(z_flat)

pca = PCA(n_components=128)
pca.fit(np.concatenate(latent_buffer))

# ----------------- 2. RUN BENCHMARK -----------------
print("Running benchmark...")

attack_names = [
    "Clean (No Attack)",
    "Gaussian Noise", 
    "Speed Change (1.1x)", 
    "Low-Pass Filter",
    "PCA Amputation (Tail=0)", 
    "PCA Jamming (Tail=Noise)", 
    "Latent Rounding"
]
# Store [Detection_Prob, SNR] for each file
results = {name: [] for name in attack_names}

for f in tqdm(test_files, desc="Attacking"):
    # Load
    wav, sr = sf.read(f)
    if sr != SAMPLE_RATE: 
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    wav = wav.astype(np.float32)
    wav_t = torch.from_numpy(wav).float().to(DEVICE).unsqueeze(0).unsqueeze(0)
    
    # Watermark
    with torch.no_grad():
        wm = wm_generator.get_watermark(wav_t, SAMPLE_RATE)
        wm_audio_t = wav_t + wm
        wm_audio = wm_audio_t.squeeze().cpu().numpy()
        
    # --- BASELINES ---
    
    # 0. Clean Watermarked
    p = detect(wm_audio)
    s = calculate_snr(wav, wm_audio) # SNR vs Original Clean
    results["Clean (No Attack)"].append((p, s))

    # 1. Gaussian Noise (Standard Black Box)
    noise_sigma = 0.01 # Tuned to be audible but not destroying content
    noise = np.random.normal(0, noise_sigma, wm_audio.shape)
    att_noise = wm_audio + noise
    p = detect(att_noise)
    s = calculate_snr(wav, att_noise)
    results["Gaussian Noise"].append((p, s))
    
    # 2. Speed Change (Resampling)
    # Resample 24k -> 22k -> 24k (shifts pitch/speed slightly)
    import librosa
    att_speed = librosa.resample(wm_audio, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE*0.95))
    att_speed = librosa.resample(att_speed, orig_sr=int(SAMPLE_RATE*0.95), target_sr=SAMPLE_RATE)
    # Fix length mismatch for detection
    if len(att_speed) < len(wm_audio):
        att_speed = np.pad(att_speed, (0, len(wm_audio)-len(att_speed)))
    else:
        att_speed = att_speed[:len(wm_audio)]
    p = detect(att_speed)
    s = calculate_snr(wav, att_speed)
    results["Speed Change (1.1x)"].append((p, s))

    # 3. Low Pass Filter (Simulate MP3/Bandwidth limit)
    # Simple moving average
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    att_lpf = np.convolve(wm_audio, kernel, mode='same')
    p = detect(att_lpf)
    s = calculate_snr(wav, att_lpf)
    results["Low-Pass Filter"].append((p, s))

    # --- SUBSPACE ATTACKS ---
    
    # Get Latents
    z = get_latents(wm_audio)
    z_T_128 = z.squeeze(0).permute(1, 0).cpu().numpy()
    z_pca = pca.transform(z_T_128)

    # 4. Amputation
    z_amp = z_pca.copy()
    z_amp[:, K_COMPONENTS:] = 0
    z_recon_amp = pca.inverse_transform(z_amp)
    z_recon_amp_t = torch.from_numpy(z_recon_amp).permute(1, 0).unsqueeze(0).to(DEVICE)
    att_amp = decode_latents(z_recon_amp_t)
    p = detect(att_amp)
    s = calculate_snr(wav, att_amp)
    results["PCA Amputation (Tail=0)"].append((p, s))

    # 5. Jamming
    z_jam = z_pca.copy()
    tail_std = np.std(z_jam[:, K_COMPONENTS:])
    z_jam[:, K_COMPONENTS:] = np.random.normal(0, tail_std * 2.0, z_jam[:, K_COMPONENTS:].shape)
    z_recon_jam = pca.inverse_transform(z_jam)
    z_recon_jam_t = torch.from_numpy(z_recon_jam).permute(1, 0).unsqueeze(0).to(DEVICE)
    att_jam = decode_latents(z_recon_jam_t)
    p = detect(att_jam)
    s = calculate_snr(wav, att_jam)
    results["PCA Jamming (Tail=Noise)"].append((p, s))

    # 6. Rounding
    scale = 5.0
    z_round = np.round(z_T_128 * scale) / scale
    z_round_t = torch.from_numpy(z_round).permute(1, 0).unsqueeze(0).to(DEVICE)
    att_round = decode_latents(z_round_t)
    p = detect(att_round)
    s = calculate_snr(wav, att_round)
    results["Latent Rounding"].append((p, s))


# ----------------- 3. VISUALIZATION -----------------

# Prepare Data
labels = list(results.keys())
mean_probs = [np.mean([x[0] for x in results[k]]) for k in labels]
mean_snrs = [np.mean([x[1] for x in results[k]]) for k in labels]

# Create Figure
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar Plot for Detection Probability (Left Axis)
x = np.arange(len(labels))
width = 0.35
rects1 = ax1.bar(x - width/2, mean_probs, width, label='Detection Prob', color='skyblue', edgecolor='black')

ax1.set_ylabel('Watermark Detection Probability', color='blue', fontsize=12)
ax1.set_ylim(0, 1.1)
ax1.tick_params(axis='y', labelcolor='blue')
ax1.axhline(0.5, color='gray', linestyle='--', label="Random Guess")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)

# Line Plot for Audio Quality SNR (Right Axis)
ax2 = ax1.twinx()
ax2.plot(x, mean_snrs, color='red', marker='o', linewidth=2, label='Audio SNR (dB)')
ax2.set_ylabel('Audio Quality (SNR dB)', color='red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='red')
# Typical good audio is >20dB
ax2.set_ylim(0, 40) 

# Title & Layout
plt.title(f"Comparison: Subspace Attacks vs Baselines (K={K_COMPONENTS})")
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_chart.png"))

print("\n--- RESULTS ---")
print(f"{'Attack':<25} | {'Det Prob':<10} | {'SNR (dB)':<10}")
print("-" * 50)
for k in labels:
    prob = np.mean([x[0] for x in results[k]])
    snr = np.mean([x[1] for x in results[k]])
    print(f"{k:<25} | {prob:.4f}     | {snr:.2f}")

print(f"\nSaved benchmark chart to {OUTPUT_DIR}")