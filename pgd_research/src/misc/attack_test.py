import os
import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from audioseal import AudioSeal
import julius

# -------- CONFIG ----------
SAMPLE_RATE = 16000 # Attack directly at 16k to avoid resampling artifacts in gradient
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 100       # Optimization steps
EPSILON = 0.02      # Max noise amplitude (Lower = Higher SNR)
ALPHA = 0.002       # Step size per iteration
LIBRISPEECH_DIR = "./wav_24k" # Will resample to 16k on load
OUTPUT_DIR = "./wm_adversarial_attack"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------

# Add this to your imports/setup
torch.backends.cudnn.enabled = False
print(f"Running PGD Attack on {DEVICE}...")

# Load Models
# We only need the generator (to create targets) and detector (to attack)
wm_generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
# Set detector to eval, but we will need gradients w.r.t Input
detector.eval() 

# ----------------- Helpers -----------------

def load_wav_16k(path):
    wav, sr = sf.read(path)
    if wav.ndim > 1: wav = np.mean(wav, axis=1)
    
    # Resample to 16k for the attack
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        
    wav = wav.astype(np.float32)
    # Norm
    wav = np.clip(wav, -1.0, 1.0)
    return wav

def calculate_snr(clean, degraded):
    noise = clean - degraded
    power_clean = np.mean(clean**2)
    power_noise = np.mean(noise**2)
    if power_noise == 0: return 100.0
    return 10 * np.log10(power_clean / power_noise)

# ----------------- ATTACK LOOP -----------------

files = [os.path.join(r, f) for r, _, fs in os.walk(LIBRISPEECH_DIR) for f in fs if f.endswith('.wav')]
test_files = files[:10] # Test on 10 files first

results_prob = []
results_snr = []

for f in tqdm(test_files, desc="Adversarial Attack"):
    # 1. Prepare Watermarked Audio
    wav_np = load_wav_16k(f)
    wav_t = torch.from_numpy(wav_np).float().to(DEVICE).unsqueeze(0).unsqueeze(0) #(1,1,T)
    
    with torch.no_grad():
        watermark = wm_generator.get_watermark(wav_t, sample_rate=16000)
        # Create the "Target" watermarked audio
        wm_audio = wav_t + watermark
    
    # 2. Setup Adversarial Variable
    # We want to find a perturbation 'delta'
    delta = torch.zeros_like(wm_audio, requires_grad=True)
    
    # 3. Optimization Loop (PGD)
    optimizer = torch.optim.Adam([delta], lr=ALPHA)
    
    for step in range(N_STEPS):
        # Apply noise
        attacked_audio = wm_audio + delta
        
        # Pass through detector
        # Detector outputs: (Batch, 2+nbits, Time)
        # Index 1 is "Watermark Probability", Index 0 is "No Watermark"
        result, _ = detector(attacked_audio, sample_rate=16000)
        
        # We want to MINIMIZE the probability of class 1 (Watermarked)
        # We take the mean over time
        prob_wm = result[:, 1, :].mean()
        
        # Loss: We want prob_wm to go to 0.
        # So we Minimize prob_wm directly.
        loss = prob_wm
        
        optimizer.zero_grad()
        # Enable training mode for backward pass (fixes cudnn RNN backward issue)
        detector.train()
        loss.backward()
        detector.eval()
        
        # Update delta
        # Using Sign-based update (FGSM style) inside PGD often works better for Linf attacks
        # But Adam is fine for general minimization. Let's add explicit clipping.
        delta.data = delta.data - ALPHA * delta.grad.sign()
        
        # Constraint: Projection (Keep noise small)
        delta.data = torch.clamp(delta.data, -EPSILON, EPSILON)
        delta.grad.zero_()
        
    # 4. Final Evaluation
    with torch.no_grad():
        final_audio = wm_audio + delta
        res, _ = detector(final_audio, sample_rate=16000)
        
        # Calculate detection score
        detect_prob = (torch.count_nonzero(torch.gt(res[:, 1, :], 0.5)) / res.shape[-1]).item()
        
        # Calculate SNR
        wm_np = wm_audio.squeeze().cpu().numpy()
        final_np = final_audio.squeeze().cpu().numpy()
        snr = calculate_snr(wm_np, final_np)
        
        results_prob.append(detect_prob)
        results_snr.append(snr)

# ----------------- VISUALIZATION -----------------

avg_prob = np.mean(results_prob)
avg_snr = np.mean(results_snr)

print("\n--- ADVERSARIAL ATTACK RESULTS ---")
print(f"Average Detection Probability: {avg_prob:.4f} (Target: 0.0)")
print(f"Average SNR: {avg_snr:.2f} dB (Target: >30)")

# Scatter Plot
plt.figure(figsize=(6, 6))
plt.scatter(results_snr, results_prob, color='red', alpha=0.7)
plt.axhline(0.5, color='gray', linestyle='--')
plt.xlabel("Audio Quality (SNR dB)")
plt.ylabel("Detection Probability")
plt.title(f"PGD Attack Performance (Epsilon={EPSILON})")
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "pgd_results.png"))
print(f"Saved plot to {OUTPUT_DIR}")