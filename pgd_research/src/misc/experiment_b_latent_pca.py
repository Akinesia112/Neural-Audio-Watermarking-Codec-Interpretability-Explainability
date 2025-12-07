import os
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import EncodecModel, AutoProcessor
from audioseal import AudioSeal
import julius

# -------- CONFIG ----------
SAMPLE_RATE = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FILES = 50  # Number of files to test
COMPONENTS_TO_TEST = [10, 20, 50, 80, 100, 128] # EnCodec dim is usually 128
LIBRISPEECH_DIR = "./wav_24k" 
OUTPUT_DIR = "./wm_pca_attack_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------

print(f"Running on {DEVICE}...")

# 1. Load Models
print("Loading models...")
# AudioSeal (16k internal, but handles resampling)
wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)

# EnCodec (24k)
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(DEVICE)
encodec.eval()

# ----------------- Helpers -----------------

def load_and_prep_wav(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1: wav = np.mean(wav, axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    wav = wav.astype(np.float32)
    # Norm to [-1, 1]
    max_val = np.max(np.abs(wav))
    if max_val > 0: wav = wav / max_val
    return wav

def get_continuous_latents(wav_np):
    """Get the continuous (dense) latents from EnCodec encoder."""
    inputs = processor(raw_audio=wav_np, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        # Encoder output: (Batch, Channels, Time) -> (1, 128, T)
        enc_out = encodec.encoder(inputs["input_values"])
        
        if hasattr(enc_out, "last_hidden_state"):
            z = enc_out.last_hidden_state
        else:
            z = enc_out[0]
    return z # Keep as Tensor for easier decoding later

def decode_latents_to_audio(latents):
    """
    Decode continuous latents directly through the decoder.
    The quantizer module in HF EnCodec is not callable (no forward method),
    so we bypass it and feed latents directly to the decoder.
    """
    with torch.no_grad():
        # Ensure correct shape
        if latents.dim() == 2:
            latents = latents.unsqueeze(0)
        if latents.dim() == 3 and latents.shape[1] != encodec.config.audio_channels:
            # EnCodec expects (B, C, T)
            latents = latents

        # Direct decode (skip quantizer)
        audio_out = encodec.decoder(latents)

        # Handle HF output object
        if hasattr(audio_out, "audio_values"):
            audio = audio_out.audio_values
        else:
            audio = audio_out

    return audio.squeeze().cpu().numpy()

# ----------------- MAIN EXPERIMENT -----------------

# 1. Collect Data
wav_files = []
for root, _, files in os.walk(LIBRISPEECH_DIR):
    for f in files:
        if f.endswith(".wav"):
            wav_files.append(os.path.join(root, f))
wav_files = sorted(wav_files)[:N_FILES]
print(f"Found {len(wav_files)} files.")

# 2. Train PCA on CLEAN Audio Latents
# We need to learn the "Elephant" (Content) shape, so we train on clean audio.
print("Training PCA on clean audio latents...")
latent_buffer = []

for path in tqdm(wav_files, desc="Extracting Clean Latents"):
    wav = load_and_prep_wav(path)
    z = get_continuous_latents(wav) # (1, 128, T)
    # Permute to (T, 128) for PCA (samples=Time, features=Channels)
    z_flat = z.squeeze(0).permute(1, 0).cpu().numpy() 
    
    # Subsample time to save memory if needed
    if z_flat.shape[0] > 500:
        z_flat = z_flat[:500, :]
    latent_buffer.append(z_flat)

X_train = np.concatenate(latent_buffer, axis=0) # (Total_Frames, 128)
print(f"PCA Training Data Shape: {X_train.shape}")

pca = PCA(n_components=128) # Keep full rank for now
pca.fit(X_train)

# Plot Explained Variance to confirm "Elephant" theory
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title("Cumulative Variance of Clean Audio Latents")
plt.xlabel("Components")
plt.ylabel("Variance Explained")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "clean_pca_variance.png"))
print("Saved clean PCA plot.")


# 3. The Attack Loop
results = {k: [] for k in COMPONENTS_TO_TEST}

print("Running PCA Reconstruction Attack on Watermarked Audio...")
for path in tqdm(wav_files, desc="Attacking"):
    # A. Prepare Watermarked Audio
    wav = load_and_prep_wav(path)
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE) #(1,1,T)
    if wav_t.dim() == 4: wav_t = wav_t.squeeze(2)

    with torch.no_grad():
        # Generate WM
        watermark = wm_generator.get_watermark(wav_t, SAMPLE_RATE)
        # Add WM
        wm_audio_t = wav_t + watermark
        wm_audio_np = wm_audio_t.squeeze().cpu().numpy()
    
    # B. Get WM Latents
    z_wm = get_continuous_latents(wm_audio_np) # (1, 128, T)
    z_wm_np = z_wm.squeeze(0).permute(1, 0).cpu().numpy() # (T, 128)

    # C. Project to PCA Space
    z_pca = pca.transform(z_wm_np) # (T, 128)

    # D. Reconstruct with top K components
    for k in COMPONENTS_TO_TEST:
        # 1. Zero out components > k
        z_pca_truncated = z_pca.copy()
        z_pca_truncated[:, k:] = 0 
        
        # 2. Inverse Transform
        z_recon_np = pca.inverse_transform(z_pca_truncated)
        
        # 3. Reshape back to Tensor for EnCodec: (1, 128, T)
        z_recon_t = torch.from_numpy(z_recon_np).permute(1, 0).unsqueeze(0).to(DEVICE)
        
        # 4. Decode to Audio (Passes through Quantizer!)
        attacked_audio = decode_latents_to_audio(z_recon_t)
        
        # 5. Detect Watermark
        attacked_audio_t = torch.from_numpy(attacked_audio).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # AudioSeal detector expects 16k usually, ensure handling
        prob, msg = detector.detect_watermark(attacked_audio_t, sample_rate=SAMPLE_RATE)
        
        results[k].append(prob)

# 4. Analyze Results
avg_scores = [np.mean(results[k]) for k in COMPONENTS_TO_TEST]

plt.figure(figsize=(8, 6))
plt.plot(COMPONENTS_TO_TEST, avg_scores, marker='o', color='red', linewidth=2)
plt.title("Attack Success: Detection vs PCA Components Kept")
plt.xlabel("Number of Principal Components Kept (Content)")
plt.ylabel("Watermark Detection Probability")
plt.ylim(0, 1.1)
plt.grid(True)
plt.axhline(y=0.5, color='gray', linestyle='--', label="Random Guess")
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "pca_attack_result.png"))

print("Results:", dict(zip(COMPONENTS_TO_TEST, avg_scores)))
print(f"Plots saved to {OUTPUT_DIR}")