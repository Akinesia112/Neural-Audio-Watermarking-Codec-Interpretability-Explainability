#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import json
from typing import List, Dict
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../interpretability_research/scripts
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # .../Neural-Audio-Watermarking-Codec-Interpretability-Explainability
WATERMARK_SRC = os.path.join(REPO_ROOT, "watermark_research", "src")
sys.path.insert(0, WATERMARK_SRC)

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm

from snac import SNAC

# ====== 1. 基本設定 ======

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_SR = 24000   # hubertsiuzdak/snac_24khz
MAX_SEC = 5       # 每段最多切到 5 秒，避免太長


def load_mono(audio_path: str, max_sec: int = MAX_SEC):
    wav, sr = torchaudio.load(audio_path)      # (C, T)
    # 多聲道 → 單聲道
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # 限制長度
    max_len = sr * max_sec
    if wav.size(-1) > max_len:
        wav = wav[..., :max_len]
    return wav, sr


# ====== 2. Watermark wrapper（這裡示範只用 AudioSeal，你可以照 pattern 加 SemanticPCA 等） ======

from watermark_testing import (
    AudioSealWM,
    WavMarkWM,
    SilentCipherWM,
    SemanticPCAWM,
    SemanticClusterWM,
    SemanticWM,
)

WM_CLASS_MAP = {
    # 傳統 / non-semantic 水印
    "AudioSeal": AudioSealWM,
    "WavMark": WavMarkWM,
    "SilentCipher": SilentCipherWM,
    # SNAC-based semantic 水印
    "SemanticPCA": SemanticPCAWM,
    "SemanticCluster": SemanticClusterWM,
    "SemanticRandom": SemanticWM,
}



def build_watermarkers(names: List[str]):
    wms = []
    for n in names:
        if n not in WM_CLASS_MAP:
            print(f"[WARN] Unknown watermark method: {n}")
            continue
        wms.append(WM_CLASS_MAP[n](DEVICE))
    return wms


# ====== 3. SNAC 層級 ablation attacker ======

class SNACLayerAblation:
    """
    給一段 audio，做：
      - resample → SNAC_SR
      - encode 得到 codes(list)
      - 對 codes 做不同 layer 的「保留 / 清零」
      - decode → 回 waveform，再 resample 回原始 sr
    """

    def __init__(self, device=DEVICE):
        self.device = device
        print(f"[SNAC] Loading hubertsiuzdak/snac_24khz on {device} ...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    @torch.inference_mode()
    def attack_with_mask(self, audio: torch.Tensor, sr_in: int,
                         layer_mask: List[int]) -> torch.Tensor:
        """
        layer_mask: list of 0/1，長度 = len(codes)
          例如 [1,1,1,1] → 用全部 codebook (baseline)
               [1,0,0,0] → 只保留最 coarse，那些 layer，其他設為 0
        """
        # 1) resample 到 SNAC_SR
        if sr_in != SNAC_SR:
            audio_24k = torchaudio.functional.resample(audio, sr_in, SNAC_SR)
        else:
            audio_24k = audio

        # SNAC 需要 (B,1,T)
        if audio_24k.dim() == 2:
            audio_24k = audio_24k.unsqueeze(0)  # (1, C, T)
        if audio_24k.size(1) > 1:
            audio_24k = audio_24k.mean(dim=1, keepdim=True)
        # 現在 (1,1,T)
        audio_24k = audio_24k.to(self.device)

        # 2) encode
        codes = self.model.encode(audio_24k)  # list of (B, T_i)

        # 3) 套用 mask：被 mask 掉的 codebook 全部設為 0
        masked_codes = []
        for code, m in zip(codes, layer_mask):
            if m == 1:
                masked_codes.append(code)
            else:
                masked_codes.append(torch.zeros_like(code))
        # 4) decode
        audio_hat_24k = self.model.decode(masked_codes)  # (B,1,T)
        audio_hat_24k = audio_hat_24k.squeeze(0)         # (1,T)

        # 5) resample 回原 sr
        if sr_in != SNAC_SR:
            audio_hat = torchaudio.functional.resample(audio_hat_24k, SNAC_SR, sr_in)
        else:
            audio_hat = audio_hat_24k

        return audio_hat.cpu()


def generate_masks(num_layers: int) -> Dict[str, List[int]]:
    """
    產生一組實驗用 mask：
      - full: [1,1,1,1]
      - drop_last_k: 從最後一層開始清零
      - keep_only_k: 只留某一層
    """
    masks = {}
    full = [1] * num_layers
    masks["full_all"] = full

    # 逐步 drop fine layers
    for k in range(1, num_layers + 1):
        m = full.copy()
        # 把最後 k 個 set 0
        for i in range(num_layers - k, num_layers):
            m[i] = 0
        masks[f"drop_last_{k}"] = m

    # 只留單一 layer
    for i in range(num_layers):
        m = [0] * num_layers
        m[i] = 1
        masks[f"keep_only_{i}"] = m

    return masks


# ====== 4. 主實驗流程 ======

def run_snac_layer_ablation(audio_dir: str,
                            out_csv: str,
                            watermark_names: List[str],
                            max_files: int = 20):
    device = DEVICE
    snac_attacker = SNACLayerAblation(device=device)
    watermarkers = build_watermarkers(watermark_names)

    files = sorted(
        glob.glob(os.path.join(audio_dir, "*.wav"))
        + glob.glob(os.path.join(audio_dir, "*.mp3"))
    )[:max_files]

    results = []

    print(f"[INFO] Found {len(files)} files.")
    for path in tqdm(files):
        fname = os.path.basename(path)
        wav, sr = load_mono(path)

        for wm in watermarkers:
            # 1. 嵌 watermark
            try:
                wm_audio, payload = wm.embed(wav, sr)
                wm_sr = wm.wm_sr
            except Exception as e:
                print(f"[WM-EMBED-ERROR] {wm.name} on {fname}: {e}")
                continue

            # NOTE: SNAC 只支援 1ch，這裡再保險一次
            if wm_audio.size(0) > 1:
                wm_audio = wm_audio.mean(dim=0, keepdim=True)

            # 2. 用 SNAC encode 取得 code 數量
            with torch.inference_mode():
                tmp_in = torchaudio.functional.resample(
                    wm_audio, wm_sr, SNAC_SR).to(device)
                if tmp_in.dim() == 2:
                    tmp_in = tmp_in.unsqueeze(0)
                if tmp_in.size(1) > 1:
                    tmp_in = tmp_in.mean(dim=1, keepdim=True)
                codes = snac_attacker.model.encode(tmp_in)
                num_layers = len(codes)

            masks = generate_masks(num_layers)

            # 3. 針對每個 mask 做 decode + detect
            for mask_name, mask in masks.items():
                try:
                    attacked = snac_attacker.attack_with_mask(
                        wm_audio, wm_sr, mask
                    )
                    score = wm.detect(attacked, wm_sr, payload)
                except Exception as e:
                    print(f"[ERROR] {wm.name}/{mask_name} on {fname}: {e}")
                    score = float("nan")

                results.append({
                    "file": fname,
                    "wm": wm.name,
                    "mask": mask_name,
                    "mask_vec": json.dumps(mask),
                    "score": score,
                })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved to {out_csv}")
    print(df.groupby(["wm", "mask"]).score.mean().sort_values(ascending=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--watermarks", nargs="+",
                        default=["AudioSeal", "SemanticPCA"])
    parser.add_argument("--filecount", type=int, default=20)
    args = parser.parse_args()

    run_snac_layer_ablation(
        audio_dir=args.audio_dir,
        out_csv=args.out_csv,
        watermark_names=args.watermarks,
        max_files=args.filecount,
    )
