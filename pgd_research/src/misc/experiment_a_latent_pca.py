import os
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import EncodecModel, AutoProcessor
from audioseal import AudioSeal

# -------- CONFIG ----------
SAMPLE_RATE = 24000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_PAIRS = 200
LIBRISPEECH_DIR = "./wav_24k" 
OUTPUT_DIR = "./wm_experiment_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------

# Load models
print("Loading AudioSeal generator & detector...")
wm_generator = AudioSeal.load_generator("audioseal_wm_16bits")
wm_generator.to(DEVICE)

print("Loading EnCodec (HF) model + processor...")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(DEVICE)
encodec.eval()

# ----------------- Helper functions -----------------

def load_wav_mono(path, target_sr=SAMPLE_RATE):
    wav, sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    # Simple resampling check
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        
    if np.issubdtype(wav.dtype, np.integer):
        wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
    else:
        wav = wav.astype(np.float32)
    wav = np.clip(wav, -1.0, 1.0)
    
    # Ensure no NaNs in raw audio
    wav = np.nan_to_num(wav)
    return wav[None, None, :]  # (1,1,T)

def get_encodec_continuous_latents(wav_np):
    """
    Extracts CONTINUOUS embeddings (before quantization).
    This is much better for PCA/Metrics than discrete codes.
    """
    # 1. Flatten for processor
    if wav_np.ndim == 3:
        wav_1d = wav_np.squeeze()
    else:
        wav_1d = wav_np

    # 2. Input Safety
    wav_1d = np.nan_to_num(wav_1d)
    wav_1d = np.clip(wav_1d, -1.0, 1.0)

    inputs = processor(raw_audio=wav_1d, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    
    with torch.no_grad():
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        # USE .encoder() NOT .encode()
        # .encoder() gives the dense float vectors [Batch, Channels, Frames]
        # .encode() gives quantized integers
        encoder_outputs = encodec.encoder(inputs["input_values"])
        
        # The encoder outputs might be a tensor or a tuple depending on version
        # Usually it returns `last_hidden_state` or the tensor directly
        if isinstance(encoder_outputs, tuple):
             lat = encoder_outputs[0]
        elif hasattr(encoder_outputs, "last_hidden_state"):
            lat = encoder_outputs.last_hidden_state
        else:
            lat = encoder_outputs

        lat = lat.detach().cpu().numpy()

    # 3. Post-process
    lat = np.atleast_2d(lat)
    lat = lat.astype(np.float32) # Ensure float32
    lat = np.nan_to_num(lat) # Clean NaNs
    
    if lat.ndim == 3: # [1, C, T] -> [C, T]
        lat = lat.squeeze(0)
        
    return lat

# ----------------- Main Loop -----------------
wav_files = []
for root, _, files in os.walk(LIBRISPEECH_DIR):
    for f in files:
        if f.endswith(".wav"):
            wav_files.append(os.path.join(root, f))
wav_files = sorted(wav_files)[:N_PAIRS]

metrics_l2 = []
metrics_cosine = []
metrics_snr = []

# Buffers for PCA
deltas_buffer = []

for path in tqdm(wav_files, desc="Processing"):
    try:
        wav = load_wav_mono(path, SAMPLE_RATE)
        wav_t = torch.from_numpy(wav).float().to(DEVICE)
        
        # AudioSeal expects 3 dims [B, C, T]
        if wav_t.dim() == 4: wav_t = wav_t.squeeze(2)

        # 1. Generate Watermark
        with torch.no_grad():
            watermark = wm_generator.get_watermark(wav_t, SAMPLE_RATE)
            wm_audio_t = wav_t + watermark
            wm_audio = wm_audio_t.squeeze(0).cpu().numpy()

        # 2. Get Continuous Latents
        clean_lat = get_encodec_continuous_latents(wav.squeeze())
        wm_lat = get_encodec_continuous_latents(wm_audio.squeeze())

        # Align lengths
        min_t = min(clean_lat.shape[-1], wm_lat.shape[-1])
        clean_lat = clean_lat[..., :min_t]
        wm_lat = wm_lat[..., :min_t]

        # 3. Calculate Delta
        delta = wm_lat - clean_lat
        
        # --- METRIC 1: L2 Norm (Energy of the watermark) ---
        # Flatten to calculate global norm per file
        d_flat = delta.flatten()
        c_flat = clean_lat.flatten()
        
        l2_val = np.linalg.norm(d_flat)
        metrics_l2.append(l2_val)

        # --- METRIC 2: Cosine Similarity ---
        # Are changes aligned with the content?
        # dot(A, B) / (|A|*|B|)
        norm_c = np.linalg.norm(c_flat)
        if norm_c > 0 and l2_val > 0:
            cos_val = np.dot(c_flat, d_flat) / (norm_c * l2_val)
        else:
            cos_val = 0.0
        metrics_cosine.append(cos_val)

        # --- METRIC 3: Latent SNR ---
        # 20 * log10( Signal / Noise )
        if l2_val > 1e-9:
            snr_val = 20 * np.log10(norm_c / l2_val)
        else:
            snr_val = 100.0 # Arbitrary high value if no noise
        metrics_snr.append(snr_val)

        # Prepare for PCA
        # We take a slice to keep memory manageable
        slice_len = 500 # Just take first 500 frames per file
        if d_flat.shape[0] > slice_len:
            deltas_buffer.append(d_flat[:slice_len])
        else:
            # Pad if too short
            padded = np.pad(d_flat, (0, slice_len - d_flat.shape[0]))
            deltas_buffer.append(padded)
            
    except Exception as e:
        print(f"Skipping {path}: {e}")

# ----------------- VISUALIZATION -----------------

# 1. Plot Metric Distributions
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].hist(metrics_l2, bins=30, color='skyblue', edgecolor='black')
axs[0].set_title("L2 Norm of Watermark (Latent Space)")
axs[0].set_xlabel("L2 Magnitude")

axs[1].hist(metrics_cosine, bins=30, color='salmon', edgecolor='black')
axs[1].set_title("Cosine Similarity (Content vs Watermark)")
axs[1].set_xlabel("Cosine Sim (-1 to 1)")
# Expected: Near 0 (Orthogonal)

axs[2].hist(metrics_snr, bins=30, color='lightgreen', edgecolor='black')
axs[2].set_title("Latent SNR (dB)")
axs[2].set_xlabel("Decibels")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_distributions.png"))
print("Saved metrics plot.")

# 2. Run PCA on the Buffer
print("Running PCA on collected deltas...")
deltas_np = np.stack(deltas_buffer)
pca = PCA(n_components=10)
pca.fit(deltas_np)
exp_var = pca.explained_variance_ratio_

plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), np.cumsum(exp_var), marker='o')
plt.title("PCA Cumulative Variance (Continuous Latents)")
plt.xlabel("Component")
plt.ylabel("Variance Ratio")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "pca_continuous.png"))
print(f"Top 5 explained variance: {exp_var[:5]}")