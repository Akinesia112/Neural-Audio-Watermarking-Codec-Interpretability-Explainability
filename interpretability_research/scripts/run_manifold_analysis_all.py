#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified manifold analysis for 6 watermarks:
  Semantic:      SemanticPCA, SemanticCluster, SemanticRandom
  Non-semantic:  AudioSeal, WavMark, SilentCipher

For each watermark:
  - get a 1D "manifold axis" in some latent space
      * semantic:   its own SNAC encoder + manifold_vector
      * nonsemantic: SNAC encoder + PCA over (wm - clean) deltas
  - project clean / watermarked / Qwen-attacked audio onto that axis
  - log summary stats and plot per-file triplets + global summary.
"""

import os
import glob
import sys
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from snac import SNAC
from watermark_testing import (
    AudioSealWM,
    WavMarkWM,
    SilentCipherWM,
    SemanticPCAWM,
    SemanticClusterWM,
    SemanticWM,
    QwenOmniAttack,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SNAC_SR = 24000
MAX_SEC = 5


def load_mono(path: str, max_sec: int = MAX_SEC):
    wav, sr = torchaudio.load(path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    max_len = sr * max_sec
    if wav.size(-1) > max_len:
        wav = wav[..., :max_len]
    return wav, sr


WM_CLASS_MAP = {
    "AudioSeal": AudioSealWM,
    "WavMark": WavMarkWM,
    "SilentCipher": SilentCipherWM,
    "SemanticPCA": SemanticPCAWM,
    "SemanticCluster": SemanticClusterWM,
    "SemanticRandom": SemanticWM,
}
SEMANTIC_NAMES = {"SemanticPCA", "SemanticCluster", "SemanticRandom"}


def build_watermarkers(names: List[str]):
    wms = []
    for n in names:
        if n not in WM_CLASS_MAP:
            print(f"[WARN] {n} is not a supported watermark (ignored).")
            continue
        wms.append(WM_CLASS_MAP[n](DEVICE))
    return wms


class SNACProbe:
    def __init__(self, device=DEVICE):
        self.device = device
        print(f"[SNACProbe] Loading hubertsiuzdak/snac_24khz on {device} ...")
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()

    @torch.inference_mode()
    def get_latent(self, audio: torch.Tensor, sr_in: int) -> torch.Tensor:
        if sr_in != SNAC_SR:
            x = torchaudio.functional.resample(audio, sr_in, SNAC_SR)
        else:
            x = audio

        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        if x.size(1) > 1:
            x = x.mean(dim=1, keepdim=True)

        x = x.to(self.device)
        z = self.model.encoder(x)[0]
        if z.dim() == 2:
            z = z.unsqueeze(0)
        return z[0].cpu()  # (D, T)


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

    print(f"[AXIS] Estimating SNAC-PCA axis for nonsemantic WM {wm_name} ...")

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

        # 這裡先檢查 delta 有沒有 NaN/Inf
        if not torch.isfinite(delta).all():
            print(f"[AXIS-WARN] non-finite delta for {wm_name} on {fname}, skip this file.")
            continue

        deltas.append(delta)
        used += 1

    if len(deltas) == 0:
        print(f"[AXIS] No valid samples for {wm_name}, skip axis.")
        return None

    deltas = torch.cat(deltas, dim=0)  # (N, D)

    # 再 double-check 一次，避免任何奇怪的 non-finite 漏過來
    finite_mask = torch.isfinite(deltas).all(dim=1)
    if finite_mask.sum() == 0:
        print(f"[AXIS] All deltas non-finite for {wm_name}, skip axis.")
        return None
    if finite_mask.sum() < deltas.size(0):
        print(f"[AXIS] Filtering out {int((~finite_mask).sum())} non-finite rows for {wm_name}.")
    deltas = deltas[finite_mask]

    deltas_centered = deltas - deltas.mean(dim=0, keepdim=True)  # (N, D)

    try:
        # SVD: deltas_centered = U S Vh, Vh: (D, D)
        U, S, Vh = torch.linalg.svd(deltas_centered, full_matrices=False)
    except RuntimeError as e:
        print(f"[AXIS-ERR] SVD failed for {wm_name}: {e}")
        return None

    axis = Vh[0]  # (D,)
    axis = axis / (axis.norm() + 1e-8)
    axis = axis.unsqueeze(1)  # (D, 1)

    print(f"[AXIS] Done for {wm_name}, used {used} files (after filtering).")
    return axis.cpu()



def plot_proj_triplet(save_dir: str,
                      base_name: str,
                      proj_clean: np.ndarray,
                      proj_wm: np.ndarray,
                      proj_attacked: np.ndarray,
                      wm_name: str,
                      wm_type: str):
    os.makedirs(save_dir, exist_ok=True)

    t_clean = np.arange(len(proj_clean))
    t_wm = np.arange(len(proj_wm))
    t_att = np.arange(len(proj_attacked))

    plt.figure(figsize=(10, 4))
    plt.plot(t_clean, proj_clean, label="clean", alpha=0.7)
    plt.plot(t_wm, proj_wm, label="watermarked", alpha=0.7)
    plt.plot(t_att, proj_attacked, label="attacked", alpha=0.7)
    plt.legend()
    plt.title(f"[{wm_type}] Projection along WM axis ({wm_name})")
    plt.xlabel("latent time index")
    plt.ylabel("projection value")
    plt.tight_layout()

    out_path = os.path.join(save_dir, f"{base_name}_proj_{wm_name}.png")
    plt.savefig(out_path)
    plt.close()


@torch.inference_mode()
def project_semantic_with_wm(wm, audio: torch.Tensor, sr_in: int, device: str):
    # 用 semantic watermark 自己的 SNAC + manifold_vector
    x = torchaudio.functional.resample(audio, sr_in, wm.wm_sr)
    if x.dim() == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 2:
        x = x.unsqueeze(0)
    if x.size(1) > 1:
        x = x.mean(dim=1, keepdim=True)

    if x.shape[-1] % 4096 != 0:
        pad = 4096 - (x.shape[-1] % 4096)
        x = torch.nn.functional.pad(x, (0, pad))

    x = x.to(device)
    z = wm.get_projected_z(x)        # (B, D, T_latent)
    manifold_vec = wm.manifold_vector.to(device)   # (D,1)
    proj = torch.matmul(z.permute(0, 2, 1), manifold_vec).squeeze(-1)
    return proj.squeeze(0).cpu()


def plot_global_summary_all(df: pd.DataFrame, out_root: str, dataset_name: str):
    if df.empty:
        print("[GLOBAL] DataFrame empty, skip global plots.")
        return

    os.makedirs(out_root, exist_ok=True)

    # ΔWM vs ΔATT scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    for (wm_name, wm_type), sub in df.groupby(["wm", "wm_type"]):
        label = f"{wm_name} ({wm_type[0]})"  # S/N short tag
        ax.scatter(sub["delta_wm"], sub["delta_att"],
                   alpha=0.6, label=label, s=20)
    ax.axhline(0.0, color="gray", linewidth=1, linestyle="--")
    ax.axvline(0.0, color="gray", linewidth=1, linestyle="--")
    xs = np.linspace(df["delta_wm"].min(), df["delta_wm"].max(), 100)
    ax.plot(xs, -xs, color="black", linewidth=1, linestyle=":")
    ax.set_xlabel("ΔWM = proj_wm_mean - proj_clean_mean")
    ax.set_ylabel("ΔATT = proj_att_mean - proj_wm_mean")
    ax.set_title(f"ΔWM vs ΔATT (All WMs, {dataset_name})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_delta = os.path.join(out_root, f"manifold_global_delta_{dataset_name}.png")
    plt.savefig(out_delta)
    plt.close()
    print(f"[GLOBAL] Saved ΔWM/ΔATT scatter to {out_delta}")

    # 3D scatter: clean / wm / att
    fig = plt.figure(figsize=(7, 6))
    ax3d = fig.add_subplot(111, projection="3d")
    for (wm_name, wm_type), sub in df.groupby(["wm", "wm_type"]):
        label = f"{wm_name} ({wm_type[0]})"
        ax3d.scatter(sub["proj_clean_mean"],
                     sub["proj_wm_mean"],
                     sub["proj_att_mean"],
                     alpha=0.6,
                     label=label,
                     s=20)
    ax3d.set_xlabel("proj_clean_mean")
    ax3d.set_ylabel("proj_wm_mean")
    ax3d.set_zlabel("proj_att_mean")
    ax3d.set_title(f"Manifold (clean→wm→att) (All WMs, {dataset_name})")
    ax3d.legend(fontsize=8)
    plt.tight_layout()
    out_3d = os.path.join(out_root, f"manifold_global_3d_{dataset_name}.png")
    plt.savefig(out_3d)
    plt.close()
    print(f"[GLOBAL] Saved 3D manifold scatter to {out_3d}")

    # bar summary per WM
    try:
        stats = df.groupby("wm")[["delta_wm", "delta_att"]].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(stats.index))
        width = 0.35
        ax.bar(x - width/2, stats["delta_wm"], width, label="ΔWM")
        ax.bar(x + width/2, stats["delta_att"], width, label="ΔATT")
        ax.set_xticks(x)
        ax.set_xticklabels(stats.index, rotation=20)
        ax.set_ylabel("mean projection difference")
        ax.set_title(f"Mean Δ per watermark ({dataset_name})")
        ax.legend()
        plt.tight_layout()
        out_bar = os.path.join(out_root, f"manifold_global_delta_bar_{dataset_name}.png")
        plt.savefig(out_bar)
        plt.close()
        print(f"[GLOBAL] Saved Δ bar plot to {out_bar}")
    except Exception:
        pass


def run_manifold_analysis_all(
    audio_dir: str,
    out_csv: str,
    watermark_names: List[str],
    max_files: int = 20,
    max_files_for_axis: int = 40,
):
    device = DEVICE
    print(f"[MAIN] device = {device}")

    wms = build_watermarkers(watermark_names)
    if len(wms) == 0:
        print("[ERROR] No valid watermark methods requested, abort.")
        return

    qwen_attacker = QwenOmniAttack(device)
    snac_probe = SNACProbe(device)

    files = sorted(
        glob.glob(os.path.join(audio_dir, "*.wav"))
        + glob.glob(os.path.join(audio_dir, "*.mp3"))
    )
    if max_files is not None:
        files = files[:max_files]
    print(f"[INFO] Found {len(files)} files to analyze.")

    out_root = os.path.dirname(out_csv)
    dataset_name = os.path.basename(os.path.normpath(audio_dir))
    plots_root = os.path.join(out_root, f"manifold_plots_{dataset_name}")
    os.makedirs(plots_root, exist_ok=True)

    # nonsemantic axis dict
    axis_dict: Dict[str, torch.Tensor] = {}
    for wm in wms:
        if wm.name in SEMANTIC_NAMES:
            continue
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
            print(f"[WARN] Skip {wm.name} in nonsemantic axis estimation.")

    rows = []

    for path in tqdm(files, desc="files"):
        fname = os.path.basename(path)
        try:
            wav_clean, sr = load_mono(path)
        except Exception as e:
            print(f"[WARN] load fail on {fname}: {e}")
            continue

        for wm in wms:
            wm_type = "semantic" if wm.name in SEMANTIC_NAMES else "nonsemantic"

            # for nonsemantic WMs we require an axis; if missing, skip
            if wm_type == "nonsemantic" and wm.name not in axis_dict:
                continue

            try:
                wm_audio, payload = wm.embed(wav_clean, sr)
                wm_sr = wm.wm_sr
            except Exception as e:
                print(f"[EMBED-ERR] {wm.name} on {fname}: {e}")
                continue

            if wm_audio.dim() == 2 and wm_audio.size(0) > 1:
                wm_audio = wm_audio.mean(dim=0, keepdim=True)

            try:
                wav_att = qwen_attacker.attack(wm_audio, wm_sr)
            except Exception as e:
                print(f"[ATTACK-ERR] {wm.name} on {fname}: {e}")
                continue

            # detection scores
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

            # projection
            try:
                if wm_type == "semantic":
                    proj_clean = project_semantic_with_wm(wm, wav_clean, sr, device)
                    proj_wm = project_semantic_with_wm(wm, wm_audio, wm_sr, device)
                    proj_att = project_semantic_with_wm(wm, wav_att, wm_sr, device)
                else:
                    axis = axis_dict[wm.name]  # (D,1)
                    z_clean = snac_probe.get_latent(wav_clean, sr)
                    z_wm = snac_probe.get_latent(wm_audio, wm_sr)
                    z_att = snac_probe.get_latent(wav_att, wm_sr)
                    proj_clean = torch.matmul(z_clean.T, axis).squeeze(-1)
                    proj_wm = torch.matmul(z_wm.T, axis).squeeze(-1)
                    proj_att = torch.matmul(z_att.T, axis).squeeze(-1)
            except Exception as e:
                print(f"[PROJ-ERR] {wm.name} on {fname}: {e}")
                continue

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
                    "wm_type": wm_type,
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

            # per-file plot
            base = os.path.splitext(fname)[0]
            wm_plot_dir = os.path.join(plots_root, wm.name)
            plot_proj_triplet(
                save_dir=wm_plot_dir,
                base_name=base,
                proj_clean=pc,
                proj_wm=pw,
                proj_attacked=pa,
                wm_name=wm.name,
                wm_type=wm_type,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(rows) == 0:
        print("[WARN] No rows collected, nothing saved.")
        return

    df = pd.DataFrame(rows)
    df["delta_wm"] = df["proj_wm_mean"] - df["proj_clean_mean"]
    df["delta_att"] = df["proj_att_mean"] - df["proj_wm_mean"]

    df.to_csv(out_csv, index=False)
    print(f"[DONE] Saved unified manifold CSV to {out_csv}")

    try:
        summary = df.groupby("wm")[
            ["proj_clean_mean", "proj_wm_mean", "proj_att_mean",
             "delta_wm", "delta_att"]
        ].agg(["mean", "std"])
        print(summary)
    except Exception:
        pass

    plot_global_summary_all(df, out_root=plots_root, dataset_name=dataset_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unified manifold analysis for 6 watermarks (semantic + nonsemantic)."
    )
    parser.add_argument("--audio_dir", type=str, required=True,
                        help="Directory containing input audio (.wav / .mp3)")
    parser.add_argument("--out_csv", type=str, required=True,
                        help="Output CSV path (directory will be created if needed)")
    parser.add_argument("--watermarks", nargs="+",
                        default=["AudioSeal", "WavMark", "SilentCipher",
                                 "SemanticPCA", "SemanticCluster", "SemanticRandom"],
                        help="Subset of the 6 supported WMs")
    parser.add_argument("--filecount", type=int, default=120,
                        help="Max number of audio files to analyze")
    parser.add_argument("--axis_files", type=int, default=120,
                        help="Max number of files used to estimate nonsemantic WM axes")

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

    run_manifold_analysis_all(
        audio_dir=args.audio_dir,
        out_csv=out_csv_abs,
        watermark_names=args.watermarks,
        max_files=args.filecount,
        max_files_for_axis=args.axis_files,
    )
