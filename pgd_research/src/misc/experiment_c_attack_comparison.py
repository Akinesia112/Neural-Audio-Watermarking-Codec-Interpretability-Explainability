import os
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from audioseal import AudioSeal
from transformers import EncodecModel, AutoProcessor
import librosa

# -------- CONFIGURATION ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE_ATTACK = 16000 # AudioSeal operates here
SAMPLE_RATE_LATENT = 24000 # EnCodec/PCA operates here

# Dataset Params
N_FILES_TRAIN = 40  # Files to learn PCA basis
N_FILES_TEST = 15   # Files to run the comparison
LIBRISPEECH_DIR = "./wav_24k" 
OUTPUT_DIR = "./wm_pgd_vs_subspace"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PCA Params
K_HEAD = 60         # Components to PROTECT (The "Elephant")

# Attack Params
N_STEPS = 60        # Optimization steps
ALPHA = 0.003       # Step size
EPSILON = 0.04      # Max noise amplitude (constraint)
PROJ_INTERVAL = 5   # How often to project to nullspace (every N steps)
# ---------------------------------

print(f"Running Experiment C on {DEVICE}...")

# 1. LOAD MODELS
print("Loading AudioSeal...")
wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
detector.eval() # We will force gradients manually

print("Loading EnCodec (for Subspace Constraint)...")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(DEVICE)
encodec.eval()

# ----------------- HELPER FUNCTIONS -----------------

def get_latents(wav_np):
    """Extract continuous latents from EnCodec."""
    if len(wav_np) == 0: return None
    # Ensure correct shape for processor
    if wav_np.ndim > 1: wav_np = wav_np.squeeze()
    
    inputs = processor(raw_audio=wav_np, sampling_rate=SAMPLE_RATE_LATENT, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        z = encodec.encoder(inputs["input_values"])
        if hasattr(z, "last_hidden_state"): z = z.last_hidden_state
        else: z = z[0]
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

# ----------------- PCA TRAINING -----------------

def train_pca():
    print("Training PCA on clean audio...")
    files = [os.path.join(r, f) for r, _, fs in os.walk(LIBRISPEECH_DIR) for f in fs if f.endswith('.wav')]
    if not files:
        raise ValueError("No wav files found!")
        
    latent_buffer = []
    
    for f in tqdm(files[:N_FILES_TRAIN], desc="PCA Train"):
        wav, sr = sf.read(f)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE_LATENT: 
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE_LATENT)
        wav = wav.astype(np.float32)
        
        # Clip to prevent artifacts
        wav = np.clip(wav, -1.0, 1.0)
        
        z = get_latents(wav)
        z_flat = z.squeeze(0).permute(1, 0).cpu().numpy() # (T, 128)
        
        # Subsample to keep memory usage low
        if z_flat.shape[0] > 500: z_flat = z_flat[:500]
        latent_buffer.append(z_flat)

    pca = PCA(n_components=128)
    pca.fit(np.concatenate(latent_buffer, axis=0))
    print(f"PCA Trained. Explained Variance of Head ({K_HEAD}): {np.sum(pca.explained_variance_ratio_[:K_HEAD]):.4f}")
    return pca

# ----------------- PROJECTION LOGIC -----------------

def project_noise_to_nullspace(delta_16k, pca, clean_16k):
    """
    Project the attack vector 'delta' into the PCA Null Space.
    Logic:
    1. Upsample noise to 24k.
    2. Measure latent difference (Non-linear projection).
    3. Zero out the 'Head' components in PCA space.
    4. Decode back to audio and downsample to 16k.
    """
    # 1. Upsample
    delta_np = delta_16k.detach().cpu().numpy().squeeze()
    clean_np = clean_16k.detach().cpu().numpy().squeeze()
    
    # Handle batch dim if present
    if delta_np.ndim == 0: return delta_16k # Safety
    
    delta_24k = librosa.resample(delta_np, orig_sr=SAMPLE_RATE_ATTACK, target_sr=SAMPLE_RATE_LATENT)
    clean_24k = librosa.resample(clean_np, orig_sr=SAMPLE_RATE_ATTACK, target_sr=SAMPLE_RATE_LATENT)
    
    # 2. Get Latents (Differential)
    attacked_24k = np.clip(clean_24k + delta_24k, -1.0, 1.0)
    
    z_clean = get_latents(clean_24k)     # (1, 128, T)
    z_attacked = get_latents(attacked_24k) # (1, 128, T)
    
    # Align lengths
    min_t = min(z_clean.shape[-1], z_attacked.shape[-1])
    z_clean = z_clean[..., :min_t]
    z_attacked = z_attacked[..., :min_t]
    
    z_diff = z_attacked - z_clean 
    z_diff_flat = z_diff.squeeze(0).permute(1, 0).cpu().numpy() # (T, 128)
    
    # 3. PCA Filtering
    z_diff_pca = pca.transform(z_diff_flat)
    
    # --- CORE CONSTRAINT: PROTECT THE HEAD ---
    z_diff_pca[:, :K_HEAD] = 0.0 
    # ---------------------------------------
    
    z_diff_recon = pca.inverse_transform(z_diff_pca)
    
    # 4. Decode (Reconstruct the filtered noise)
    z_target = z_clean + torch.from_numpy(z_diff_recon).permute(1, 0).unsqueeze(0).to(DEVICE)
    audio_recon_24k = decode_latents(z_target)
    
    # 5. Downsample back to 16k
    audio_recon_16k = librosa.resample(audio_recon_24k, orig_sr=SAMPLE_RATE_LATENT, target_sr=SAMPLE_RATE_ATTACK)
    
    # Fix length mismatch
    target_len = delta_16k.shape[-1]
    if len(audio_recon_16k) > target_len:
        audio_recon_16k = audio_recon_16k[:target_len]
    elif len(audio_recon_16k) < target_len:
        audio_recon_16k = np.pad(audio_recon_16k, (0, target_len - len(audio_recon_16k)))
        
    # The new filtered delta
    new_delta_np = audio_recon_16k - clean_np
    
    return torch.from_numpy(new_delta_np).float().to(DEVICE).view_as(delta_16k)

# ----------------- MAIN COMPARISON -----------------

# Train PCA
pca = train_pca()

# Prepare Test Set
files = [os.path.join(r, f) for r, _, fs in os.walk(LIBRISPEECH_DIR) for f in fs if f.endswith('.wav')]
test_files = files[N_FILES_TRAIN : N_FILES_TRAIN + N_FILES_TEST]

results = {
    "Standard PGD": {"prob": [], "snr": []},
    "Subspace PGD": {"prob": [], "snr": []}
}

print("\n--- STARTING ATTACK COMPARISON ---")

for f in tqdm(test_files, desc="Attacking"):
    # Load and prep (16k)
    wav_np = sf.read(f)[0]
    if len(wav_np.shape) > 1: wav_np = wav_np.mean(axis=1)
    if sf.read(f)[1] != SAMPLE_RATE_ATTACK:
        wav_np = librosa.resample(wav_np, orig_sr=sf.read(f)[1], target_sr=SAMPLE_RATE_ATTACK)
    wav_np = np.clip(wav_np, -1.0, 1.0)
    
    # Create Tensor
    wav_t = torch.from_numpy(wav_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0)

    # Generate Watermark (Target)
    with torch.no_grad():
        wm = wm_generator.get_watermark(wav_t, sample_rate=SAMPLE_RATE_ATTACK)
        wm_audio = wav_t + wm

    # Run Comparison
    modes = ["Standard PGD", "Subspace PGD"]
    
    for mode in modes:
        # Init Noise
        delta = torch.zeros_like(wm_audio, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=ALPHA)
        
        # IMPORTANT: Disable CuDNN for RNN backward compatibility in Eval mode
        with torch.backends.cudnn.flags(enabled=False):
            for step in range(N_STEPS):
                attacked_audio = wm_audio + delta
                
                # Get detection probability
                res, _ = detector(attacked_audio, sample_rate=SAMPLE_RATE_ATTACK)
                prob_wm = res[:, 1, :].mean()
                
                # Loss = Detection Prob (Minimize it)
                loss = prob_wm 
                
                optimizer.zero_grad()
                loss.backward()
                
                # Update (Sign gradient)
                delta.data = delta.data - ALPHA * delta.grad.sign()
                
                # --- SUBSPACE PROJECTION (The key difference) ---
                if mode == "Subspace PGD" and (step + 1) % PROJ_INTERVAL == 0:
                    with torch.no_grad():
                        filtered_delta = project_noise_to_nullspace(delta, pca, wm_audio)
                        delta.data = filtered_delta
                
                # Standard Clipping
                delta.data = torch.clamp(delta.data, -EPSILON, EPSILON)
                delta.grad.zero_()
        
        # --- FINAL METRICS ---
        with torch.no_grad():
            final_audio = wm_audio + delta
            res, _ = detector(final_audio, sample_rate=SAMPLE_RATE_ATTACK)
            
            # 1. Detection Score (Mean probability of being watermarked)
            # We use soft probability for granularity
            final_prob = res[:, 1, :].mean().item()
            
            # 2. SNR
            noise_np = (final_audio - wm_audio).squeeze().cpu().numpy()
            clean_np = wm_audio.squeeze().cpu().numpy()
            
            power_clean = np.mean(clean_np**2)
            power_noise = np.mean(noise_np**2)
            
            if power_noise == 0: snr = 100.0
            else: snr = 10 * np.log10(power_clean / power_noise)
            
            results[mode]["prob"].append(final_prob)
            results[mode]["snr"].append(snr)

# ----------------- VISUALIZATION -----------------

print("\n--- FINAL RESULTS SUMMARY ---")
plt.figure(figsize=(10, 7))

colors = {"Standard PGD": "red", "Subspace PGD": "green"}
markers = {"Standard PGD": "x", "Subspace PGD": "o"}

for mode in results:
    probs = results[mode]["prob"]
    snrs = results[mode]["snr"]
    
    avg_prob = np.mean(probs)
    avg_snr = np.mean(snrs)
    print(f"{mode}: Det Prob = {avg_prob:.4f} | Avg SNR = {avg_snr:.2f} dB")
    
    plt.scatter(snrs, probs, color=colors[mode], marker=markers[mode], 
                label=f'{mode} (Avg SNR: {avg_snr:.1f}dB)', alpha=0.7, s=80)

plt.axhline(0.5, color='gray', linestyle='--', label="Random Guess (0.5)")
plt.xlabel("Audio Quality (SNR dB) -> Higher is Better")
plt.ylabel("Watermark Detection Probability -> Lower is Better")
plt.title("Attack Comparison: Standard vs. Subspace-Constrained PGD")
plt.legend()
plt.grid(True, alpha=0.3)

out_path = os.path.join(OUTPUT_DIR, "experiment_c_results.png")
plt.savefig(out_path)
print(f"Results saved to {out_path}")