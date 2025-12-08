#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import sys
from typing import List

THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../interpretability_research/scripts
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # .../Neural-Audio-Watermarking-Codec-Interpretability-Explainability
WATERMARK_SRC = os.path.join(REPO_ROOT, "watermark_research", "src")
sys.path.insert(0, WATERMARK_SRC)

import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from snac import SNAC
from watermark_testing import (
    SemanticPCAWM, SemanticClusterWM, SemanticWM, QwenOmniAttack
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_SR = 24000
MAX_SEC = 5


def load_mono(path):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_len = sr * MAX_SEC
    if wav.size(-1) > max_len:
        wav = wav[..., :max_len]
    return wav, sr


def build_semantic_wms(device, names: List[str]):
    """只建 Semantic 系列，names 用來 filter。"""
    name_map = {
        "SemanticPCA": SemanticPCAWM,
        "SemanticCluster": SemanticClusterWM,
        "SemanticRandom": SemanticWM,
    }
    wms = []
    for n in names:
        if n not in name_map:
            print(f"[WARN] {n} is not a semantic watermark (ignored in manifold analysis).")
            continue
        wms.append(name_map[n](device))
    if not wms:
        # fallback: 全部 semantic
        print("[INFO] No valid semantic names given, using all Semantic* WMs.")
        wms = [SemanticPCAWM(device), SemanticClusterWM(device), SemanticWM(device)]
    return wms


class SNACProbe:
    """共用一個 SNAC 來抽 latent + 做 manifold 投影"""

    def __init__(self, device=DEVICE):
        self.device = device
        print(f"[SNACProbe] Loading hubertsiuzdak/snac_24khz on {device} ...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    @torch.inference_mode()
    def get_latent_and_proj(self, audio: torch.Tensor, sr_in: int,
                            manifold_vec: torch.Tensor):
        """
        audio: (1,T)
        manifold_vec: (D,1) on same device as model
        回傳：
          - z: (B, D, T_latent)
          - proj: (T_latent,) 在 manifold 上的投影
        """
        # 1) resample
        if sr_in != SNAC_SR:
            x = torchaudio.functional.resample(audio, sr_in, SNAC_SR)
        else:
            x = audio
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1,1,T)
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)
        x = x.to(self.device)

        # 2) encoder
        z = self.model.encoder(x)[0]  # (B, D, T_latent) or (D, T)
        if z.dim() == 2:
            z = z.unsqueeze(0)

        # 3) proj: (B,T)
        proj = torch.matmul(z.permute(0, 2, 1), manifold_vec).squeeze(-1)  # (B, T)
        return z.cpu(), proj.squeeze(0).cpu()


def plot_proj_triplet(save_dir, base_name,
                      proj_clean, proj_wm, proj_attacked,
                      wm_name):
    os.makedirs(save_dir, exist_ok=True)
    t_clean = np.arange(len(proj_clean))
    t_wm = np.arange(len(proj_wm))
    t_att = np.arange(len(proj_attacked))

    plt.figure(figsize=(10, 4))
    plt.plot(t_clean, proj_clean, label="clean", alpha=0.7)
    plt.plot(t_wm, proj_wm, label="watermarked", alpha=0.7)
    plt.plot(t_att, proj_attacked, label="attacked", alpha=0.7)
    plt.legend()
    plt.title(f"Projection along manifold ({wm_name})")
    plt.xlabel("latent time index")
    plt.ylabel("projection value")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{base_name}_proj_{wm_name}.png")
    plt.savefig(out_path)
    plt.close()


def run_semantic_manifold_analysis(audio_dir: str,
                                   out_csv: str,
                                   watermark_names: List[str],
                                   max_files: int = 20):
    device = DEVICE
    semantic_wms = build_semantic_wms(device, watermark_names)
    qwen_attacker = QwenOmniAttack(device)
    snac_probe = SNACProbe(device)

    files = sorted(
        glob.glob(os.path.join(audio_dir, "*.wav"))
        + glob.glob(os.path.join(audio_dir, "*.mp3"))
    )[:max_files]

    rows = []
    out_plot_dir = os.path.join(os.path.dirname(out_csv), "manifold_plots")
    os.makedirs(out_plot_dir, exist_ok=True)

    print(f"[INFO] Found {len(files)} files.")

    for path in tqdm(files):
        fname = os.path.basename(path)
        wav_clean, sr = load_mono(path)

        for wm in semantic_wms:
            # 1. embed
            try:
                wav_wm, payload = wm.embed(wav_clean, sr)
            except Exception as e:
                print(f"[EMBED-ERR] {wm.name} on {fname}: {e}")
                continue

            # 2. attack (Qwen Omni)
            try:
                wav_att = qwen_attacker.attack(wav_wm, wm.wm_sr)
            except Exception as e:
                print(f"[ATTACK-ERR] {wm.name} on {fname}: {e}")
                continue

            # 3. 檢測分數
            try:
                score_clean = wm.detect(wav_clean, sr, payload=None)
            except Exception:
                score_clean = float("nan")
            score_wm = wm.detect(wav_wm, wm.wm_sr, payload)
            score_att = wm.detect(wav_att, wm.wm_sr, payload)

            # 4. latent & projection
            manifold_vec = wm.manifold_vector.to(device)
            _, proj_clean = snac_probe.get_latent_and_proj(wav_clean, sr, manifold_vec)
            _, proj_wm = snac_probe.get_latent_and_proj(wav_wm, wm.wm_sr, manifold_vec)
            _, proj_att = snac_probe.get_latent_and_proj(wav_att, wm.wm_sr, manifold_vec)

            def stats(x):
                return {
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }

            stat_clean = stats(proj_clean)
            stat_wm = stats(proj_wm)
            stat_att = stats(proj_att)

            rows.append({
                "file": fname,
                "wm": wm.name,
                "score_clean": score_clean,
                "score_wm": score_wm,
                "score_attacked": score_att,
                "proj_clean_mean": stat_clean["mean"],
                "proj_clean_std": stat_clean["std"],
                "proj_wm_mean": stat_wm["mean"],
                "proj_wm_std": stat_wm["std"],
                "proj_att_mean": stat_att["mean"],
                "proj_att_std": stat_att["std"],
            })

            base = os.path.splitext(fname)[0]
            plot_proj_triplet(
                os.path.join(out_plot_dir, wm.name),
                base,
                proj_clean.numpy(),
                proj_wm.numpy(),
                proj_att.numpy(),
                wm.name
            )

        # 每個 file 結束後順手清一下 cache，避免 CUDA 撐太高
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] saved CSV to {out_csv}")
    if not df.empty:
        print(df.groupby("wm")[["proj_clean_mean", "proj_wm_mean", "proj_att_mean"]].mean())
    else:
        print("[WARN] No rows collected — maybe all embeddings/attacks failed?")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze semantic manifold projections of watermark embeddings under Qwen/SNAC."
    )
    parser.add_argument(
        "--audio_dir", type=str, required=True,
        help="Directory containing input audio files (.wav / .mp3)"
    )
    parser.add_argument(
        "--out_csv", type=str, required=True,
        help="Path to output CSV (will be created; parent dir auto-created)"
    )
    parser.add_argument(
        "--watermarks", nargs="+",
        default=["SemanticPCA", "SemanticCluster", "SemanticRandom"],
        help="Which semantic watermarks to analyze (SemanticPCA / SemanticCluster / SemanticRandom)"
    )
    parser.add_argument(
        "--filecount", type=int, default=20,
        help="Max number of audio files to analyze"
    )

    args = parser.parse_args()

    out_csv_abs = os.path.abspath(args.out_csv)
    out_dir = os.path.dirname(out_csv_abs)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[MAIN] audio_dir = {args.audio_dir}")
    print(f"[MAIN] out_csv   = {out_csv_abs}")
    print(f"[MAIN] watermarks= {args.watermarks}")
    print(f"[MAIN] filecount = {args.filecount}")

    run_semantic_manifold_analysis(
        audio_dir=args.audio_dir,
        out_csv=out_csv_abs,
        watermark_names=args.watermarks,
        max_files=args.filecount,
    )
