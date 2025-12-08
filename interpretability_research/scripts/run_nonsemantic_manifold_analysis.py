#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-semantic manifold analysis for traditional watermarks:
  - AudioSeal  (diffusion-style WM)
  - WavMark    (time-domain spread spectrum WM)
  - SilentCipher (codec-based speech WM)

Idea:
  1) Use SNAC encoder as a generic audio representation.
  2) For each watermark method, collect latent differences:
       delta = mean_z(wm_audio) - mean_z(clean_audio)
     and run PCA to get the main "watermark direction" in SNAC latent space.
  3) For each file, project clean / watermarked / Qwen-attacked audio
     onto this direction, and log summary stats + time-series plots.
"""

import os
import glob
import sys
import json
from typing import List, Dict

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
    AudioSealWM,
    WavMarkWM,
    SilentCipherWM,
    QwenOmniAttack,
)

# ====== 1. 基本設定 ======

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_SR = 24000
MAX_SEC = 5  # 每段 audio 最多取 5 秒避免太長及 OOM


def load_mono(path: str, max_sec: int = MAX_SEC):
    """Load audio, mix to mono, truncate to <= max_sec seconds."""
    wav, sr = torchaudio.load(path)  # (C, T)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_len = sr * max_sec
    if wav.size(-1) > max_len:
        wav = wav[..., :max_len]
    return wav, sr


# ====== 2. Watermark wrapper ======

WM_CLASS_MAP = {
    "AudioSeal": AudioSealWM,
    "WavMark": WavMarkWM,
    "SilentCipher": SilentCipherWM,
}


def build_watermarkers(names: List[str]):
    wms = []
    for n in names:
        if n not in WM_CLASS_MAP:
            print(f"[WARN] {n} is not a supported non-semantic watermark (ignored).")
            continue
        wms.append(WM_CLASS_MAP[n](DEVICE))
    return wms


# ====== 3. SNAC Probe (generic latent extractor) ======

class SNACProbe:
    """
    A lightweight wrapper to use SNAC as a generic audio encoder.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        print(f"[SNACProbe] Loading hubertsiuzdak/snac_24khz on {device} ...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    @torch.inference_mode()
    def get_latent(self, audio: torch.Tensor, sr_in: int) -> torch.Tensor:
        """
        audio: (1, T) on CPU
        return: z (D, T_latent) on CPU
        """
        # resample
        if sr_in != SNAC_SR:
            x = torchaudio.functional.resample(audio, sr_in, SNAC_SR)
        else:
            x = audio

        # (1, T) -> (B, 1, T)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)  # (1, C, T)
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)

        x = x.to(self.device)
        z = self.model.encoder(x)[0]  # (B, D, T) or (D, T)

        if z.dim() == 2:
            z = z.unsqueeze(0)
        # return first batch as (D, T)
        return z[0].cpu()


# ====== 4. PCA-based "watermark manifold direction" ======

def estimate_axis_for_wm(
    wm_name: str,
    wm_obj,
    files: List[str],
    snac_probe: SNACProbe,
    max_files_for_axis: int = 40,
) -> torch.Tensor:
    """
    For a given watermark method:
      - take up to max_files_for_axis audio files
      - embed watermark
      - encode clean & wm in SNAC latent space
      - compute delta = mean_z(wm) - mean_z(clean)
      - run PCA over all deltas to get principal direction.

    Returns:
      axis: (D, 1) tensor on CPU (unit vector) or None if fail.
    """
    deltas = []
    used = 0

    print(f"[AXIS] Estimating manifold axis for {wm_name} ...")

    for path in files:
        if used >= max_files_for_axis:
            break
        fname = os.path.basename(path)

        try:
            wav_clean, sr = load_mono(path)
        except Exception as e:
            print(f"[AXIS-WARN] load fail on {fname}: {e}")
            continue

        try:
            wm_audio, payload = wm_obj.embed(wav_clean, sr)
            wm_sr = wm_obj.wm_sr
        except Exception as e:
            print(f"[AXIS-WARN] embed fail ({wm_name}) on {fname}: {e}")
            continue

        try:
            # SNAC latent
            z_clean = snac_probe.get_latent(wav_clean, sr)       # (D, T1)
            z_wm = snac_probe.get_latent(wm_audio, wm_sr)        # (D, T2)
        except Exception as e:
            print(f"[AXIS-WARN] SNAC encode fail ({wm_name}) on {fname}: {e}")
            continue

        # mean over time: (D,)
        mu_clean = z_clean.mean(dim=1)
        mu_wm = z_wm.mean(dim=1)
        delta = (mu_wm - mu_clean).unsqueeze(0)  # (1, D)
        deltas.append(delta)
        used += 1

    if len(deltas) == 0:
        print(f"[AXIS] No valid samples for {wm_name}, skip axis.")
        return None

    deltas = torch.cat(deltas, dim=0)  # (N, D)
    deltas_centered = deltas - deltas.mean(dim=0, keepdim=True)  # (N, D)

    # SVD: deltas_centered = U S Vh, Vh: (D, D)
    U, S, Vh = torch.linalg.svd(deltas_centered, full_matrices=False)
    axis = Vh[0]  # (D,)
    axis = axis / (axis.norm() + 1e-8)
    axis = axis.unsqueeze(1)  # (D, 1)

    print(f"[AXIS] Done for {wm_name}, used {used} files.")
    return axis.cpu()


def plot_proj_triplet_nonsemantic(
    save_dir: str,
    base_name: str,
    proj_clean: np.ndarray,
    proj_wm: np.ndarray,
    proj_attacked: np.ndarray,
    wm_name: str,
):
    os.makedirs(save_dir, exist_ok=True)

    t_clean = np.arange(len(proj_clean))
    t_wm = np.arange(len(proj_wm))
    t_att = np.arange(len(proj_attacked))

    plt.figure(figsize=(10, 4))
    plt.plot(t_clean, proj_clean, label="clean", alpha=0.7)
    plt.plot(t_wm, proj_wm, label="watermarked", alpha=0.7)
    plt.plot(t_att, proj_attacked, label="attacked", alpha=0.7)
    plt.legend()
    plt.title(f"[Non-Semantic] Projection along WM axis ({wm_name})")
    plt.xlabel("latent time index")
    plt.ylabel("projection value")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{base_name}_proj_{wm_name}.png")
    plt.savefig(out_path)
    plt.close()


# ====== 5. 主實驗流程 ======

def run_nonsemantic_manifold_analysis(
    audio_dir: str,
    out_csv: str,
    watermark_names: List[str],
    max_files: int = 20,
    max_files_for_axis: int = 40,
):
    device = DEVICE
    print(f"[MAIN] device = {device}")

    # Watermarks: AudioSeal / WavMark / SilentCipher
    wms = build_watermarkers(watermark_names)
    if len(wms) == 0:
        print("[ERROR] No valid watermark methods requested, abort.")
        return

    # Qwen / SNAC LALM attacker
    qwen_attacker = QwenOmniAttack(device)

    # SNAC probe
    snac_probe = SNACProbe(device)

    # Collect file list
    files = sorted(
        glob.glob(os.path.join(audio_dir, "*.wav"))
        + glob.glob(os.path.join(audio_dir, "*.mp3"))
    )
    if max_files is not None:
        files = files[:max_files]
    print(f"[INFO] Found {len(files)} files to analyze.")

    # 1) 為每個 WM 估計一個 non-semantic axis
    axis_dict: Dict[str, torch.Tensor] = {}
    for wm in wms:
        axis = estimate_axis_for_wm(
            wm_name=wm.name,
            wm_obj=wm,
            files=files,
            snac_probe=snac_probe,
            max_files_for_axis=max_files_for_axis,
        )
        if axis is not None:
            axis_dict[wm.name] = axis
        else:
            print(f"[WARN] Skip {wm.name} in manifold analysis (no axis).")

    if len(axis_dict) == 0:
        print("[ERROR] No WM axis estimated, nothing to do.")
        return

    # 2) 對所有檔案 / WM 做 projection + detection logging
    rows = []
    out_plot_root = os.path.join(os.path.dirname(out_csv), "nonsemantic_manifold_plots")
    os.makedirs(out_plot_root, exist_ok=True)

    for path in tqdm(files, desc="files"):
        fname = os.path.basename(path)

        try:
            wav_clean, sr = load_mono(path)
        except Exception as e:
            print(f"[WARN] load fail on {fname}: {e}")
            continue

        for wm in wms:
            if wm.name not in axis_dict:
                continue
            axis = axis_dict[wm.name]  # (D, 1) CPU

            # 2.1 embed watermark
            try:
                wm_audio, payload = wm.embed(wav_clean, sr)
                wm_sr = wm.wm_sr
            except Exception as e:
                print(f"[EMBED-ERR] {wm.name} on {fname}: {e}")
                continue

            # 保險：mono
            if wm_audio.dim() == 2 and wm_audio.size(0) > 1:
                wm_audio = wm_audio.mean(dim=0, keepdim=True)

            # 2.2 attack via Qwen/SNAC
            try:
                wav_att = qwen_attacker.attack(wm_audio, wm_sr)
            except Exception as e:
                print(f"[ATTACK-ERR] {wm.name} on {fname}: {e}")
                continue

            # 2.3 detection scores
            # clean 上 detect 常常不合理，先 best-effort（可能會 NaN）
            try:
                score_clean = wm.detect(wav_clean, sr, payload=None)
            except Exception:
                score_clean = float("nan")

            try:
                score_wm = wm.detect(wm_audio, wm_sr, payload)
            except Exception as e:
                print(f"[DETECT-ERR] wm ({wm.name}) on {fname}: {e}")
                score_wm = float("nan")

            try:
                score_att = wm.detect(wav_att, wm_sr, payload)
            except Exception as e:
                print(f"[DETECT-ERR] att ({wm.name}) on {fname}: {e}")
                score_att = float("nan")

            # 2.4 SNAC latent + projection onto WM-specific axis
            try:
                z_clean = snac_probe.get_latent(wav_clean, sr)    # (D, T_c)
                z_wm = snac_probe.get_latent(wm_audio, wm_sr)     # (D, T_w)
                z_att = snac_probe.get_latent(wav_att, wm_sr)     # (D, T_a)
            except Exception as e:
                print(f"[SNAC-ERR] {wm.name} on {fname}: {e}")
                continue

            # axis / latent on CPU
            a = axis  # (D,1)

            # proj(t) = z(t)^T a
            # z: (D, T) → (T, D) @ (D,1) = (T,1)
            proj_clean = torch.matmul(z_clean.T, a).squeeze(-1)  # (T_c,)
            proj_wm = torch.matmul(z_wm.T, a).squeeze(-1)        # (T_w,)
            proj_att = torch.matmul(z_att.T, a).squeeze(-1)      # (T_a,)

            pc = proj_clean.numpy()
            pw = proj_wm.numpy()
            pa = proj_att.numpy()

            def stats(x: np.ndarray):
                return {
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }

            stat_clean = stats(pc)
            stat_wm = stats(pw)
            stat_att = stats(pa)

            rows.append(
                {
                    "file": fname,
                    "wm": wm.name,
                    "score_clean": score_clean,
                    "score_wm": score_wm,
                    "score_attacked": score_att,
                    "proj_clean_mean": stat_clean["mean"],
                    "proj_clean_std": stat_clean["std"],
                    "proj_clean_min": stat_clean["min"],
                    "proj_clean_max": stat_clean["max"],
                    "proj_wm_mean": stat_wm["mean"],
                    "proj_wm_std": stat_wm["std"],
                    "proj_wm_min": stat_wm["min"],
                    "proj_wm_max": stat_wm["max"],
                    "proj_att_mean": stat_att["mean"],
                    "proj_att_std": stat_att["std"],
                    "proj_att_min": stat_att["min"],
                    "proj_att_max": stat_att["max"],
                }
            )

            # 2.5 畫 time-series 圖
            base = os.path.splitext(fname)[0]
            plot_dir = os.path.join(out_plot_root, wm.name)
            plot_proj_triplet_nonsemantic(
                save_dir=plot_dir,
                base_name=base,
                proj_clean=pc,
                proj_wm=pw,
                proj_attacked=pa,
                wm_name=wm.name,
            )

    if len(rows) == 0:
        print("[WARN] No rows collected, nothing saved.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved non-semantic manifold CSV to {out_csv}")

    try:
        print(
            df.groupby("wm")[
                ["proj_clean_mean", "proj_wm_mean", "proj_att_mean"]
            ].mean()
        )
    except Exception:
        pass


# ====== 6. CLI ======

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Non-semantic manifold analysis for traditional audio watermarks (AudioSeal, WavMark, SilentCipher) using SNAC latent."
    )
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing input audio (.wav / .mp3)")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV path (directory will be created if needed)")
    parser.add_argument("--watermarks", nargs="+",
                        default=["AudioSeal", "WavMark", "SilentCipher"],
                        help="Subset of [AudioSeal, WavMark, SilentCipher]")
    parser.add_argument("--filecount", type=int, default=20,
                        help="Max number of audio files to analyze")
    parser.add_argument("--axis_files", type=int, default=40,
                        help="Max number of files used to estimate WM axis")

    args = parser.parse_args()

    out_csv_abs = os.path.abspath(args.out_csv)
    out_dir = os.path.dirname(out_csv_abs)
    if out_dir != "" and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"[MAIN] audio_dir  = {args.audio_dir}")
    print(f"[MAIN] out_csv    = {out_csv_abs}")
    print(f"[MAIN] filecount  = {args.filecount}")
    print(f"[MAIN] axis_files = {args.axis_files}")
    print(f"[MAIN] watermarks = {args.watermarks}")

    run_nonsemantic_manifold_analysis(
        audio_dir=args.audio_dir,
        out_csv=out_csv_abs,
        watermark_names=args.watermarks,
        max_files=args.filecount,
        max_files_for_axis=args.axis_files,
    )
