import torch
import torchaudio
import numpy as np
import os
import torch.nn as nn
from snac import SNAC
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class ManifoldWatermarker:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading SNAC on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Determine latent dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 24000).to(device)
            z = self.model.encoder(dummy)[0]
            # FIX: Ensure 3D shape before checking dimension
            if z.dim() == 2: z = z.unsqueeze(0)
            self.latent_dim = z.shape[1] 
            
        print(f"Latent Dimension: {self.latent_dim}")
        
        # Define Manifold (Random Hyperplane)
        rng = np.random.RandomState(42)
        v_np = rng.randn(self.latent_dim).astype(np.float32)
        v_np /= np.linalg.norm(v_np)
        self.manifold_vector = torch.tensor(v_np, device=device).unsqueeze(1) # (Dim, 1)

    def measure_projection(self, audio):
        """
        Detector: Returns the average projection onto the manifold vector.
        Positive = Watermarked. Negative/Zero = Unwatermarked.
        """
        with torch.no_grad():
            if audio.dim() == 1: audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2: audio = audio.unsqueeze(0)
            
            if audio.shape[-1] % 4096 != 0:
                pad = 4096 - (audio.shape[-1] % 4096)
                audio = torch.nn.functional.pad(audio, (0, pad))
            
            # Get Continuous Latents
            z = self.model.encoder(audio)[0]
            
            # --- FIX: Handle missing batch dimension ---
            if z.dim() == 2: z = z.unsqueeze(0) # (Batch, Dim, Time)
            
            # Calculate Projection: z dot v
            # z.permute(0, 2, 1) -> (Batch, Time, Dim)
            # manifold_vector -> (Dim, 1)
            # Result -> (Batch, Time, 1)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            # Score is the mean projection across time
            score = projections.mean().item()
            return score

    def inject_manifold(self, audio, sr, steps=200, lr=0.01, epsilon=0.02, target_score=5.0):
        """
        Optimizes audio to push its latent embedding onto the positive side of the manifold.
        """
        target_sr = 24000
        wav_input = torchaudio.functional.resample(audio, sr, target_sr).to(self.device)
        
        if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
        elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
        
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))
        
        # Amplitude mask for silence (Avoid watermarking silence)
        amplitude = wav_input.abs()
        mask = (amplitude > 0.02).float() 
        
        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        print(f"Injecting Manifold Watermark ({steps} steps)...")
        pbar = tqdm(range(steps))
        
        for i in pbar:
            optimizer.zero_grad()
            
            effective_delta = delta * mask
            perturbed_audio = wav_input + effective_delta
            
            # Get Latents
            z = self.model.encoder(perturbed_audio)[0] 
            
            # --- FIX: Handle missing batch dimension ---
            if z.dim() == 2: z = z.unsqueeze(0)
            
            # Calculate Projections
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            # Loss: Hinge Loss
            # We want projection > target_score.
            # If (target - proj) is positive, we have error. If negative, we are good (loss=0).
            loss = torch.relu(target_score - projections).mean()
            
            loss.backward()
            
            delta.grad *= mask
            optimizer.step()
            
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)
            
            if i % 10 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

        final_audio = wav_input + (delta.detach() * mask)
        noise = (delta.detach() * mask)
        return final_audio.squeeze().cpu(), noise.squeeze().cpu()

def run_manifold_experiment(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        designer = ManifoldWatermarker(device)
    except Exception as e:
        print(f"Failed to init: {e}")
        return

    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[-1] > sr*5: wav = wav[:, :sr*5]
    except Exception as e:
        print(f"Load failed: {e}")
        return
    
    print("\n--- Phase 1: Checking Original ---")
    score_orig = designer.measure_projection(wav.to(device))
    print(f"Original Projection Score: {score_orig:.4f} (Expected near 0)")
    
    print("\n--- Phase 2: Injecting Manifold ---")
    # Using target_score=2.0 to ensure a strong push
    wm_audio, noise = designer.inject_manifold(wav, sr, steps=200, lr=0.01, epsilon=0.02, target_score=2.0)
    
    print("\n--- Phase 3: Verifying ---")
    score_wm = designer.measure_projection(wm_audio.to(device))
    print(f"Watermarked Projection Score: {score_wm:.4f}")
    
    torchaudio.save("manifold_wm.wav", wm_audio.unsqueeze(0), 24000)
    torchaudio.save("manifold_noise.wav", noise.unsqueeze(0), 24000)
    print("Saved output files.")

    if score_wm > 1.0:
        print("\n[SUCCESS] Audio successfully pushed onto the manifold!")
    else:
        print("\n[FAIL] Could not push audio onto manifold.")

if __name__ == "__main__":
    TEST_FILE = "/mnt/data/home/lily/SPML/wav_24k/2277-149896-0005.wav"

    run_manifold_experiment(TEST_FILE)