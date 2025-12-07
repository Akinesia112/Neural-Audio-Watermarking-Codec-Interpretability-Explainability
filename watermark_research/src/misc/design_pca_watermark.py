import torch
import torchaudio
import numpy as np
import os
import torch.nn as nn
from snac import SNAC
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class PCAWatermarker:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading SNAC on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 1. Locate Quantizer Module (Layer 0)
        self.quantizer_module = None
        if hasattr(self.model.quantizer, 'quantizers'):
            self.quantizer_module = self.model.quantizer.quantizers[0]
        else:
            raise AttributeError("Could not locate Quantizer Layer 0.")

        # 2. Locate Codebook
        self.codebook = None
        if hasattr(self.quantizer_module, 'codebook'):
             if isinstance(self.quantizer_module.codebook, nn.Embedding):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
             elif hasattr(self.quantizer_module.codebook, 'weight'):
                 self.codebook = self.quantizer_module.codebook.weight.detach()
        
        if self.codebook is None:
            raise AttributeError("Could not locate Codebook Embedding.")
            
        # 3. Locate Projection Layer (768 -> 8)
        # SNAC quantizers often have 'in_proj' or 'project_in'
        self.projector = None
        
        # Check standard names
        for attr in ['in_proj', 'project_in', 'input_conv']:
            if hasattr(self.quantizer_module, attr):
                self.projector = getattr(self.quantizer_module, attr)
                print(f"Projection Layer Found: {attr}")
                break
        
        # If still not found, check dimensions
        # Encoder dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 24000).to(device)
            z_raw = self.model.encoder(dummy)[0]
            self.enc_dim = z_raw.shape[1]
        
        self.cb_dim = self.codebook.shape[1]
        print(f"Encoder Dim: {self.enc_dim} -> Codebook Dim: {self.cb_dim}")
        
        if self.enc_dim != self.cb_dim and self.projector is None:
            print("WARNING: Dimensions mismatch but no projection layer found. Searching sub-modules...")
            # Fallback scan for Conv1d with correct out_channels
            for name, mod in self.quantizer_module.named_children():
                if isinstance(mod, nn.Conv1d) and mod.out_channels == self.cb_dim:
                    self.projector = mod
                    print(f"Projection Layer inferred: {name}")
                    break
        
        # 4. PCA Alignment
        print("Calculating PCA of Codebook...")
        cb_centered = self.codebook - self.codebook.mean(dim=0, keepdim=True)
        _, _, V = torch.linalg.svd(cb_centered)
        self.manifold_vector = V[0].unsqueeze(1) # (8, 1)

    def get_projected_z(self, audio):
        """Helper to get Z and project it down to codebook dimension"""
        # Encoder
        z = self.model.encoder(audio)[0]
        if z.dim() == 2: z = z.unsqueeze(0)
        
        # Project (768 -> 8)
        if self.projector:
            z = self.projector(z)
            
        return z

    def measure_projection(self, audio):
        with torch.no_grad():
            if audio.dim() == 1: audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2: audio = audio.unsqueeze(0)
            
            if audio.shape[-1] % 4096 != 0:
                pad = 4096 - (audio.shape[-1] % 4096)
                audio = torch.nn.functional.pad(audio, (0, pad))
            
            # Get Projected Latents
            z = self.get_projected_z(audio)
            
            # Project onto PCA Vector
            # z: (B, 8, T) -> permute -> (B, T, 8)
            # vector: (8, 1)
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            return projections.mean().item()

    def attack_lalm(self, audio, sr):
        with torch.no_grad():
            target_sr = 24000
            wav_input = torchaudio.functional.resample(audio, sr, target_sr).to(self.device)
            
            if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
            elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
            if wav_input.shape[-1] % 4096 != 0:
                pad = 4096 - (wav_input.shape[-1] % 4096)
                wav_input = torch.nn.functional.pad(wav_input, (0, pad))
                
            codes = self.model.encode(wav_input)
            rec = self.model.decode(codes)
            return rec.squeeze().cpu()

    def inject_manifold(self, audio, sr, steps=200, lr=0.01, epsilon=0.02, target_score=5.0):
        target_sr = 24000
        wav_input = torchaudio.functional.resample(audio, sr, target_sr).to(self.device)
        
        if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
        elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))
        
        amplitude = wav_input.abs()
        mask = (amplitude > 0.02).float() 
        
        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        print(f"Injecting PCA-Aligned Watermark ({steps} steps)...")
        pbar = tqdm(range(steps))
        
        for i in pbar:
            optimizer.zero_grad()
            
            effective_delta = delta * mask
            perturbed_audio = wav_input + effective_delta
            
            # Use Helper to get projected Z
            z = self.get_projected_z(perturbed_audio)
            
            # Project onto Manifold
            projections = torch.matmul(z.permute(0, 2, 1), self.manifold_vector).squeeze()
            
            # Loss
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

def run_experiment(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        designer = PCAWatermarker(device)
    except Exception as e:
        print(f"Init Failed: {e}")
        return

    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[-1] > sr*5: wav = wav[:, :sr*5]
    except Exception as e:
        print(f"Load failed: {e}")
        return
    
    print("\n--- Phase 1: Checking Original ---")
    score_orig = designer.measure_projection(wav.to(device))
    print(f"Original Projection: {score_orig:.4f}")
    
    print("\n--- Phase 2: Injecting PCA Manifold ---")
    # Using epsilon 0.02 and target 3.0
    wm_audio, noise = designer.inject_manifold(wav, sr, steps=250, lr=0.01, epsilon=0.005, target_score=0.5)
    
    print("\n--- Phase 3: Immediate Robustness Check ---")
    score_wm = designer.measure_projection(wm_audio.to(device))
    print(f"Watermarked Projection: {score_wm:.4f}")
    
    print("Performing Attack (Re-synthesis)...")
    attacked_wav = designer.attack_lalm(wm_audio, 24000)
    score_attacked = designer.measure_projection(attacked_wav.to(device))
    print(f"Post-Attack Projection: {score_attacked:.4f}")
    
    torchaudio.save("pca_wm.wav", wm_audio.unsqueeze(0), 24000)
    torchaudio.save("pca_attacked.wav", attacked_wav.unsqueeze(0), 24000)
    
    # Robustness Metric
    boost_initial = score_wm - score_orig
    boost_final = score_attacked - score_orig
    
    if boost_initial > 0:
        retention = (boost_final / boost_initial) * 100
        print(f"\nRetention: {retention:.1f}%")
        
    if score_attacked > score_orig + 0.5:
        print("[SUCCESS] The PCA-Aligned watermark survived LALM Quantization!")
    else:
        print("[FAIL] Quantization still killed it. Increase target_score or epsilon.")

if __name__ == "__main__":
    TEST_FILE = "/mnt/data/home/lily/SPML/wav_24k/2277-149896-0005.wav" 
    
    run_experiment(TEST_FILE)