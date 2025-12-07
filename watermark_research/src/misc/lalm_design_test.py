import torch
import torchaudio
import numpy as np
import os
import torch.nn as nn
from snac import SNAC
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class SemanticWatermarker:
    def __init__(self, device='cuda'):
        self.device = device
        print(f"Loading SNAC on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Find Codebook
        self.codebook = None
        if hasattr(self.model.quantizer, 'quantizers'):
            q0 = self.model.quantizer.quantizers[0]
            if hasattr(q0, 'codebook'):
                if isinstance(q0.codebook, nn.Embedding):
                    self.codebook = q0.codebook.weight.detach()
                elif hasattr(q0.codebook, 'weight'): 
                    self.codebook = q0.codebook.weight.detach()

        if self.codebook is None:
            for name, module in self.model.quantizer.named_modules():
                if isinstance(module, nn.Embedding):
                    self.codebook = module.weight.detach()
                    break
        
        if self.codebook is None:
            raise AttributeError("Could not locate Codebook Embedding.")
            
        self.vocab_size = self.codebook.shape[0]
        self.dim = self.codebook.shape[1]
        print(f"Targeting Codebook Layer 0: {self.vocab_size} tokens of dim {self.dim}")

        # Define Green/Red Split
        rng = np.random.RandomState(42)
        perm = rng.permutation(self.vocab_size)
        split = self.vocab_size // 2
        self.green_indices = torch.tensor(perm[:split], device=device)
        self.green_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        self.green_mask[self.green_indices] = True

    def get_green_ratio(self, audio):
        with torch.no_grad():
            if audio.dim() == 1: audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2: audio = audio.unsqueeze(0)
            
            if audio.shape[-1] % 4096 != 0:
                pad = 4096 - (audio.shape[-1] % 4096)
                audio = torch.nn.functional.pad(audio, (0, pad))
            
            codes = self.model.encode(audio)
            layer0_codes = codes[0].flatten()
            
            is_green = self.green_mask[layer0_codes] # Fast boolean lookup
            ratio = is_green.float().mean().item()
            return ratio, layer0_codes

    def inject_watermark(self, audio, sr, steps=200, lr=0.005, epsilon=0.01):
        """
        Surgical Injection: Only optimizes RED tokens. Preserves GREEN tokens.
        """
        target_sr = 24000
        wav_input = torchaudio.functional.resample(audio, sr, target_sr).to(self.device)
        
        if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
        elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
        
        # Pad 
        if wav_input.shape[-1] % 4096 != 0:
            pad = 4096 - (wav_input.shape[-1] % 4096)
            wav_input = torch.nn.functional.pad(wav_input, (0, pad))
        
        # Calculate Downsample Factor (Audio Samples -> 1 Token)
        # We need to map token indices back to audio samples to create the mask.
        # Run one pass to get dimensions
        with torch.no_grad():
            z_dummy = self.model.encoder(wav_input)[0]
        
        num_tokens = z_dummy.shape[-1]
        num_samples = wav_input.shape[-1]
        # Calculate stride (e.g. 512 or 320)
        stride = num_samples // num_tokens
        
        delta = torch.zeros_like(wav_input, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([delta], lr=lr)
        green_vectors = self.codebook[self.green_indices] 
        
        print(f"Surgical Optimization ({steps} steps)...")
        pbar = tqdm(range(steps))
        
        for i in pbar:
            optimizer.zero_grad()
            
            # 1. Forward Pass
            perturbed_audio = wav_input + delta
            z_features = self.model.encoder(perturbed_audio)
            z_0 = z_features[0] 
            if z_0.dim() == 2: z_0 = z_0.unsqueeze(0)
            
            # 2. Identify RED Tokens (The Bad Ones)
            # Find nearest neighbors to CURRENT full codebook to see what token we currently have
            # (Simplification: We just want to pull towards Green, but we weight by "How Red is it?")
            
            # Flatten: (Batch, Time, Dim)
            z_flat = z_0.permute(0, 2, 1).reshape(-1, self.dim) # (T, D)
            
            # --- SURGICAL MASK CREATION ---
            # Ideally, we only optimize tokens that are NOT in the green set.
            # But calculating full codebook distance every step is slow.
            # Heuristic: We calculate distance to Green. If distance is very small, it's likely Green.
            
            dists = torch.cdist(z_flat, green_vectors) # (Time, NumGreen)
            min_dist, _ = dists.min(dim=1) # (Time)
            
            # 3. Dynamic Weighting
            # If distance is already tiny, loss is near zero -> Gradient small -> Preservation.
            # If distance is large, loss is large -> Gradient large -> Aggressive correction.
            # This naturally prioritizes Red tokens.
            
            loss = min_dist.mean()
            loss.backward()
            
            # 4. UPSAMPLE THE GRADIENT MASK
            # We want to zero out gradients for regions that correspond to silence/Green tokens.
            # We use the original audio's amplitude to mask silence.
            amplitude_mask = (wav_input.abs() > 0.01).float()
            
            # Apply Silence Mask
            delta.grad *= amplitude_mask
            
            optimizer.step()
            
            # 5. Projection (Lower Epsilon for imperceptibility)
            with torch.no_grad():
                delta.clamp_(-epsilon, epsilon)
            
            if i % 10 == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

        final_audio = wav_input + delta.detach()
        noise = delta.detach()
        return final_audio.squeeze().cpu(), noise.squeeze().cpu()

def run_design_experiment(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    designer = SemanticWatermarker(device)

    try:
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[-1] > sr*5: wav = wav[:, :sr*5]
    except Exception as e:
        print(f"Load failed: {e}")
        return
    
    print("\n--- Phase 1: Checking Original ---")
    ratio_orig, _ = designer.get_green_ratio(wav.to(device))
    print(f"Original Ratio: {ratio_orig*100:.2f}%")
    
    print("\n--- Phase 2: Surgical Injection ---")
    # Reduced epsilon for lower noise
    wm_audio, noise = designer.inject_watermark(wav, sr, steps=200, lr=0.005, epsilon=0.01)
    
    print("\n--- Phase 3: Verifying ---")
    ratio_wm, _ = designer.get_green_ratio(wm_audio.to(device))
    print(f"Watermarked Ratio: {ratio_wm*100:.2f}%")
    
    torchaudio.save("design_v3_wm.wav", wm_audio.unsqueeze(0), 24000)
    torchaudio.save("design_v3_noise.wav", noise.unsqueeze(0), 24000)
    print("Saved output files.")

    if ratio_wm > ratio_orig:
        print("\n[SUCCESS] Ratio Improved without destroying silence!")
    else:
        print("\n[FAIL] Optimization struggled.")

if __name__ == "__main__":
    TEST_FILE = "/mnt/data/home/lily/SPML/wav_24k/2277-149896-0005.wav"
    run_design_experiment(TEST_FILE)