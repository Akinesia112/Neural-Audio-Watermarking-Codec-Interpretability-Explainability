import os
import torch
import soundfile as sf
import numpy as np
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Correct imports for Qwen-Audio
# ----------------------------
from transformers import Qwen2Tokenizer, Qwen2AudioForConditionalGeneration

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "Qwen/Qwen2-Audio-7B"
DEVICE = "cuda"  # or "cpu"

# ----------------------------
# Load tokenizer and model
# ----------------------------
tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
model.eval()

# ----------------------------
# LALM (Qwen-Audio) roundtrip
# ----------------------------
def lalmaudio_roundtrip_local_wav(wav_path):
    # 1. Prepare audio query
    sp_prompt = "<|startoftranscript|><|en|><|transcribe|><|en|><|notimestamps|><|wo_itn|>"
    query = f"<audio>{wav_path}</audio>{sp_prompt}"

    # 2. Process audio
    audio_info = tokenizer.process_audio(query)
    inputs = tokenizer(query, return_tensors="pt", audio_info=audio_info)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 3. Generate (encode→decode)
    with torch.no_grad():
        pred = model.generate(**inputs, audio_info=audio_info)

    # 4. Decode output
    decoded = tokenizer.decode(pred[0].cpu(), skip_special_tokens=False, audio_info=audio_info)

    # 5. Extract reconstructed waveform
    import re, base64
    from io import BytesIO
    m = re.search(r"<audio>(.*?)</audio>", decoded)
    if m is None:
        raise ValueError("No audio found in decoded output")
    audio_str = m.group(1)
    audio_bytes = base64.b64decode(audio_str)
    f = BytesIO(audio_bytes)
    rec_audio, sr = sf.read(f)
    return rec_audio, sr

# ----------------------------
# Audio similarity metric
# ----------------------------
def si_sdr(reference, estimate, eps=1e-8):
    reference = torch.tensor(reference)
    estimate = torch.tensor(estimate[:len(reference)])
    alpha = torch.sum(reference * estimate) / (torch.sum(reference ** 2) + eps)
    projection = alpha * reference
    noise = estimate - projection
    sdr = 10 * torch.log10((torch.sum(projection ** 2) + eps) / (torch.sum(noise ** 2) + eps))
    return float(sdr)

# ----------------------------
# Placeholder watermark detection
# ----------------------------
def detect_watermark(audio, method="placeholder"):
    # Replace with actual detection code
    return None

# ----------------------------
# Experiment pipeline
# ----------------------------
def evaluate_directory(input_dir, output_csv="results.csv"):
    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    results = []

    for fname in tqdm(files):
        wav_path = os.path.join(input_dir, fname)
        try:
            # Load original audio
            orig_audio, sr = sf.read(wav_path)

            # Detect watermark before
            wm_before = detect_watermark(orig_audio)

            # Roundtrip through Qwen-Audio
            rec_audio, rec_sr = lalmaudio_roundtrip_local_wav(wav_path)

            # Detect watermark after
            wm_after = detect_watermark(rec_audio)

            # Audio similarity
            sdr = si_sdr(orig_audio, rec_audio)
            mse = np.mean((orig_audio - rec_audio[:len(orig_audio)])**2)

            results.append({
                "file": fname,
                "wm_before": wm_before,
                "wm_after": wm_after,
                "watermark_removed": wm_before is not None and wm_after is None,
                "SI-SDR": sdr,
                "MSE": mse
            })

        except Exception as e:
            print(f"Error processing {fname}: {e}")
            results.append({
                "file": fname,
                "wm_before": None,
                "wm_after": None,
                "watermark_removed": None,
                "SI-SDR": None,
                "MSE": None
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return df

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/mnt/data/home/lily/SPML/wav_24k", help="Directory containing WAV files")
    parser.add_argument("--out", type=str, default="results.csv")
    args = parser.parse_args()

    evaluate_directory(args.dir, args.out)
