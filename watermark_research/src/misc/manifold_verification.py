import torch
import torchaudio
import numpy as np
import os
from snac import SNAC

class ManifoldVerifier:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading SNAC on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        self.model.eval()
        
        # --- CRITICAL: Reconstruct the EXACT same manifold vector ---
        # We must use the same random seed (42) as the design script
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 24000).to(device)
            z = self.model.encoder(dummy)[0]
            if z.dim() == 2: z = z.unsqueeze(0)
            self.latent_dim = z.shape[1]
            
        rng = np.random.RandomState(42)
        v_np = rng.randn(self.latent_dim).astype(np.float32)
        v_np /= np.linalg.norm(v_np)
        self.manifold_vector = torch.tensor(v_np, device=device).unsqueeze(1) # (Dim, 1)
        print(f"Manifold Vector Reconstructed (Dim {self.latent_dim})")

    def measure_projection(self, audio):
        with torch.no_grad():
            if audio.dim() == 1: audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2: audio = audio.unsqueeze(0)
            
            if audio.shape[-1] % 4096 != 0:
                pad = 4096 - (audio.shape[-1] % 4096)
                audio = torch.nn.functional.pad(audio, (0, pad))
            
            # Get Latents
            z = self.model.encoder(audio)[0]
            if z.dim() == 2: z = z.unsqueeze(0)
            
            # Project
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            return projections.mean().item()

    def attack_lalm(self, audio, sr):
        """
        Simulates the full LALM attack: Encoder -> Quantizer -> Decoder
        """
        with torch.no_grad():
            target_sr = 24000
            wav_input = torchaudio.functional.resample(audio, sr, target_sr).to(self.device)
            
            if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
            elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
            
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))
                
            # Full Reconstruction (The "Attack")
            codes = self.model.encode(wav_input)
            reconstructed_wav = self.model.decode(codes)
            
            return reconstructed_wav.squeeze().cpu()

def run_verification(wm_path, original_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verifier = ManifoldVerifier(device)

    # 1. Measure Watermarked Audio (Before Attack)
    print(f"\n--- Loading Watermarked Audio: {wm_path} ---")
    wav_wm, sr = torchaudio.load(wm_path)
    score_wm = verifier.measure_projection(wav_wm.to(device))
    print(f"Pre-Attack Score:  {score_wm:.4f}")

    # 2. Perform LALM Attack (Re-synthesis)
    print("\n--- Performing LALM Attack (Re-synthesis) ---")
    attacked_wav = verifier.attack_lalm(wav_wm, sr)
    
    # 3. Measure Attacked Audio
    score_attacked = verifier.measure_projection(attacked_wav.to(device))
    print(f"Post-Attack Score: {score_attacked:.4f}")
    
    # 4. Compare with Original (if provided)
    if original_path and os.path.exists(original_path):
        wav_orig, _ = torchaudio.load(original_path)
        score_orig = verifier.measure_projection(wav_orig.to(device))
        print(f"Original Score:    {score_orig:.4f}")
        
        # Calculate Robustness Ratio
        # How much of the "boost" survived?
        boost_initial = score_wm - score_orig
        boost_final = score_attacked - score_orig
        if boost_initial > 0:
            retention = (boost_final / boost_initial) * 100
            print(f"\nWatermark Retention: {retention:.1f}%")
    
    torchaudio.save("final_attacked_output.wav", attacked_wav.unsqueeze(0), 24000)
    
    # Final Verdict
    if score_attacked > 1.5: # Arbitrary robust threshold
        print("\n[SUCCESS] The watermark survived the LALM hallucination!")
    elif score_attacked > score_orig + 0.5:
        print("\n[PARTIAL SUCCESS] Watermark weakened but still detectable.")
    else:
        print("\n[FAIL] The LALM sanitized the watermark.")

if __name__ == "__main__":
    # Point to the files you just generated
    WM_FILE = "manifold_wm.wav"
    ORIG_FILE = "/mnt/data/home/lily/SPML/wav_24k/2277-149896-0005.wav" 

    if os.path.exists(WM_FILE):
        run_verification(WM_FILE, ORIG_FILE)
    else:
        print(f"Could not find {WM_FILE}. Run the design script first.")