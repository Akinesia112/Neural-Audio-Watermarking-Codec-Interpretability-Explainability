import torch
import torchaudio
import os
import numpy as np
import warnings
from snac import SNAC

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. The "Open Heart Surgery" Attacker (Fixed & Robust) ---
class QwenOmniLayerAttack:
    def __init__(self, device):
        self.device = device
        print(f"Loading SNAC for Layer Analysis on {device}...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        self.target_sr = 24000

    def attack_layer_bypass(self, audio: torch.Tensor, input_sr: int, mode="standard"):
        """
        Modes:
        - "standard": Normal LALM (Encoder -> Quantize -> Decoder)
        - "bypass_quantizer": Encoder -> (Skip Quantizer) -> Decoder
        """
        with torch.no_grad():
            # 1. Resample to SNAC rate (24k)
            wav_input = torchaudio.functional.resample(audio, input_sr, self.target_sr)
            
            # Ensure shape (Batch, Channels, Time) -> (1, 1, T)
            if wav_input.dim() == 1: wav_input = wav_input.unsqueeze(0).unsqueeze(0)
            elif wav_input.dim() == 2: wav_input = wav_input.unsqueeze(0)
            wav_input = wav_input.to(self.device)

            # --- FIX 1: Padding ---
            # Pad input to be divisible by 4096 (stride) to avoid tensor mismatch errors
            stride_multiple = 4096
            B, C, T = wav_input.shape
            remainder = T % stride_multiple
            if remainder != 0:
                pad_len = stride_multiple - remainder
                wav_input = torch.nn.functional.pad(wav_input, (0, pad_len))

            try:
                # --- FIX 2: Correct Attribute Access ---
                # A. Encoder Pass
                z = self.model.encoder(wav_input)
                
                if mode == "bypass_quantizer":
                    # --- THE BYPASS ---
                    # Pass continuous features directly to decoder
                    reconstructed_wav = self.model.decoder(z)
                    
                else: 
                    # --- STANDARD PATH ---
                    # --- FIX 3: Safe Tuple Unpacking ---
                    # SNAC quantizer returns variable items (z_q, codes, loss).
                    # We only need the first item (z_q).
                    quantizer_out = self.model.quantizer(z)
                    z_q = quantizer_out[0] 
                    
                    reconstructed_wav = self.model.decoder(z_q)

            except AttributeError as e:
                print(f"CRITICAL ERROR: Model structure mismatch. {e}")
                raise e 
            except Exception as e:
                print(f"Layer Hack Failed: {e}")
                raise e

            # 3. Cleanup & Resample Back
            reconstructed_wav = reconstructed_wav.squeeze().cpu()
            if reconstructed_wav.dim() == 0: return torch.zeros(1, input_sr)
            if reconstructed_wav.dim() == 1: reconstructed_wav = reconstructed_wav.unsqueeze(0)
            
            out_audio = torchaudio.functional.resample(reconstructed_wav, self.target_sr, input_sr)
            
            # Trim the padding we added (match original length)
            if out_audio.shape[-1] > audio.shape[-1]:
                out_audio = out_audio[..., :audio.shape[-1]]
            elif out_audio.shape[-1] < audio.shape[-1]:
                pad = audio.shape[-1] - out_audio.shape[-1]
                out_audio = torch.nn.functional.pad(out_audio, (0, pad))
                
            return out_audio

# --- 2. AudioSeal Wrapper ---
class AudioSealWM:
    def __init__(self, device):
        self.name = "AudioSeal"
        from audioseal import AudioSeal
        self.generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
        self.detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
        self.wm_sr = 16000
        self.device = device

    def embed(self, audio, sr):
        # AudioSeal works at 16k
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            watermark = self.generator.get_watermark(wav_16k, self.wm_sr)
            watermarked_audio = wav_16k + watermark
        return watermarked_audio.squeeze(0).cpu(), "audioseal_bit"

    def detect(self, audio, sr, payload):
        wav_input = torchaudio.functional.resample(audio, sr, self.wm_sr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            result, _ = self.detector.detect_watermark(wav_input, self.wm_sr)
            score = result.mean().item() if isinstance(result, torch.Tensor) else result
        return score

# --- 3. WavMark Wrapper ---
class WavMarkWM:
    def __init__(self, device):
        self.name = "WavMark"
        import wavmark
        self.model = wavmark.load_model().to(device)
        self.wm_sr = 16000
        self.device = device

    def embed(self, audio, sr):
        import wavmark
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).numpy().flatten()
        payload = np.random.choice([0, 1], size=16)
        try:
            wm_wav, _ = wavmark.encode_watermark(self.model, wav_16k, payload, show_progress=False)
            return torch.tensor(wm_wav).unsqueeze(0), payload
        except: return audio, None

    def detect(self, audio, sr, payload):
        import wavmark
        if payload is None: return 0.0
        wav_16k = torchaudio.functional.resample(audio, sr, self.wm_sr).numpy().flatten()
        try:
            decoded, _ = wavmark.decode_watermark(self.model, wav_16k, show_progress=False)
            if decoded is None: return 0.0
            return 1.0 - np.mean(payload != decoded) # Accuracy
        except: return 0.0

# --- 4. SilentCipher Wrapper (Fixed) ---
class SilentCipherWM:
    def __init__(self, device):
        self.name = "SilentCipher"
        import silentcipher
        import numpy as np
        self.np = np
        # Load model
        print("  [SilentCipher] Loading Model...")
        self.model = silentcipher.get_model(
            ckpt_path='/mnt/data/home/lily/spml-final/silentcipher-release/Models/44_1_khz/73999_iteration',
            config_path='/mnt/data/home/lily/spml-final/silentcipher-release/Models/44_1_khz/73999_iteration/hparams.yaml',
            model_type='44.1k', 
            device=device
        )
        self.wm_sr = 44100
        self.device = device
        
        # Discover correct message length once during init
        self.valid_length = self._discover_length()
        print(f"  [SilentCipher] Discovered valid message length: {self.valid_length}")

    def _discover_length(self):
        """
        Brute-force finds the expected message length by trying 1..64.
        This runs only once at startup.
        """
        # Create a dummy silent audio clip for testing
        dummy_wav = self.np.zeros(44100 * 3, dtype=self.np.float32) # 3 seconds
        
        # Try lengths 1 to 64
        for length in range(1, 65):
            try:
                msg = [1] * length # Dummy message of specific length
                # We catch stdout/stderr to suppress the "Using default SDR" print spam
                # but for simplicity, we just run it.
                self.model.encode_wav(dummy_wav, self.wm_sr, msg)
                return length # If we get here, it worked!
            except AssertionError:
                continue # Wrong length
            except Exception:
                continue # Other errors
        
        print("  [SilentCipher] WARNING: Could not auto-discover length. Defaulting to 31.")
        return 31

    def embed(self, audio, sr):
        # 1. Resample to 44.1k
        wav = torchaudio.functional.resample(audio, sr, self.wm_sr)
        
        # 2. Convert to Mono
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
        
        # 3. Pad to minimum duration (3 seconds safe margin)
        min_samples = 3 * 44100
        original_length = wav.shape[-1]
        
        if original_length < min_samples:
            pad_len = min_samples - original_length
            wav = torch.nn.functional.pad(wav, (0, pad_len))
        
        # 4. Flatten
        wav_np = wav.cpu().squeeze().numpy()
        
        # 5. Generate Payload using DISCOVERED length
        # Convert to standard python list of ints
        msg = self.np.random.randint(0, 2, size=self.valid_length).tolist()
        msg = [int(b) for b in msg] # Ensure pure python ints
        
        try:
            # 6. Encode
            enc, _ = self.model.encode_wav(wav_np, self.wm_sr, msg)
            
            # 7. Convert back to Tensor
            out = torch.tensor(enc).to(self.device).float()
            
            # 8. Unpad
            if out.shape[-1] > original_length:
                 out = out[..., :original_length]

            if out.dim() == 1: out = out.unsqueeze(0)
            return out, msg
            
        except Exception as e:
            print(f"SilentCipher Embed Error: {repr(e)}")
            return audio, None

    def detect(self, audio, sr, payload):
        wav = torchaudio.functional.resample(audio, sr, self.wm_sr)
        if wav.shape[0] > 1: wav = torch.mean(wav, dim=0, keepdim=True)
        
        # Pad for detection
        min_samples = 3 * 44100
        if wav.shape[-1] < min_samples:
             pad_len = min_samples - wav.shape[-1]
             wav = torch.nn.functional.pad(wav, (0, pad_len))

        wav_np = wav.cpu().squeeze().numpy()
        
        try:
            res = self.model.decode_wav(wav_np, self.wm_sr, phase_shift_decoding=True)
            if res and 'messages' in res:
                 for m in res['messages']:
                     if m == payload: return 1.0
            return 0.0
        except: return 0.0

# --- 5. Main Experiment Loop ---
def run_comparison(audio_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Running Standalone Layer Verification on {device} ---")
    
    # 1. Setup
    attacker = QwenOmniLayerAttack(device)
    watermarkers = [
        AudioSealWM(device), 
        WavMarkWM(device),
        SilentCipherWM(device)  # Added SilentCipher
    ]
    
    # 2. Load Audio
    try:
        wav, sr = torchaudio.load(audio_path)
        # Limit to 5s to keep it fast
        if wav.shape[-1] > sr*5: wav = wav[:, :sr*5] 
    except Exception as e:
        print(f"Could not load audio: {e}")
        return

    for wm in watermarkers:
        print(f"\n[{wm.name}] Embedding Watermark...")
        wm_audio, payload = wm.embed(wav, sr)
        
        # Sanity Check
        orig_score = wm.detect(wm_audio, wm.wm_sr, payload)
        print(f"[{wm.name}] Original Score: {orig_score:.4f}")

        if orig_score < 0.8:
            print(f"[{wm.name}] WARNING: Embedding weak/failed on original. Skipping.")
            continue

        try:
            # Test Standard
            audio_std = attacker.attack_layer_bypass(wm_audio, wm.wm_sr, mode="standard")
            score_std = wm.detect(audio_std, wm.wm_sr, payload)
            
            # Test Bypass
            audio_bypass = attacker.attack_layer_bypass(wm_audio, wm.wm_sr, mode="bypass_quantizer")
            score_bypass = wm.detect(audio_bypass, wm.wm_sr, payload)

            # Report
            print(f"[{wm.name}] Standard (Quantized): Score {score_std:.4f}")
            print(f"[{wm.name}] Bypass (Continuous):  Score {score_bypass:.4f}")
            
            # Logic for thresholds
            threshold = 0.5 if wm.name == "AudioSeal" else 0.85
            if wm.name == "SilentCipher": threshold = 0.9
            
            if score_bypass > threshold:
                print(f"--> CONCLUSION: {wm.name} survives Encoder but dies at Quantizer.")
            else:
                print(f"--> CONCLUSION: {wm.name} is destroyed by the Encoder (Resampling/Convolutions).")
                
        except Exception as e:
            print(f"[{wm.name}] CRITICAL FAIL: {e}")

if __name__ == "__main__":
    # CHANGE THIS to your real file path
    TEST_FILE = "/mnt/data/home/lily/SPML/wav_24k/2277-149896-0005.wav"
    
    run_comparison(TEST_FILE)