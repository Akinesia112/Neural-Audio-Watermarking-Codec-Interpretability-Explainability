#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        header = f.readline()
    sep = "|" if "|" in header else ","
    return pd.read_csv(path, sep=sep)


def plot_attack_metric_bars(df_attack, metric, out_path, ylabel=None):
    """bar: metric vs attack_type"""
    if "attack_type" not in df_attack.columns:
        raise ValueError("stats table needs 'attack_type' column.")

    x = np.arange(len(df_attack))
    y = df_attack[metric].values
    names = df_attack["attack_type"].tolist()

    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel(ylabel or metric)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  -> {out_path}")


def plot_snr_box_by_attack(df_results, metric, out_path):
    """
    用原始 test_results_loose.csv 畫某個 metric 在不同 attack 下的 boxplot。
    這可以看作「不同攻擊對 SNR 的整體擾動分佈」。
    """
    if "attack_type" not in df_results.columns:
        raise ValueError("results table needs 'attack_type' column.")

    attacks = sorted(df_results["attack_type"].unique())
    data = [df_results.loc[df_results["attack_type"] == a, metric].values for a in attacks]

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=attacks, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by attack_type")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  -> {out_path}")


def plot_feature_importance_bar(imp_df, out_path, top_k=20):
    """bar: |pearson| for top-K features"""
    df = imp_df.sort_values("abs_pearson", ascending=False).head(top_k)
    x = np.arange(len(df))
    y = df["abs_pearson"].values
    names = df["feature"].tolist()

    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("|pearson|")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot attack-level watermark stats & feature importance."
    )
    parser.add_argument(
        "--stats_by_attack",
        type=str,
        required=True,
        help="outputs/wm_stats_loose/stats_by_attack.csv",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="runs/.../results/test_results_loose.csv (original per-chunk results).",
    )
    parser.add_argument(
        "--feature_importance_csv",
        type=str,
        required=True,
        help="outputs/wm_stats_loose/feature_importance_by_attack.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for plots.",
    )
    parser.add_argument(
        "--top_k_features",
        type=int,
        default=20,
    )
    args = parser.parse_args()

    stats_att_path = Path(args.stats_by_attack)
    results_path = Path(args.results_csv)
    fi_path = Path(args.feature_importance_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_att = load_table(stats_att_path)
    df_res = load_table(results_path)
    df_fi = load_table(fi_path)

    # 1) attack-level performance footprint
    if "bitwise/distorted_mean" in df_att.columns:
        plot_attack_metric_bars(
            df_att,
            metric="bitwise/distorted_mean",
            out_path=out_dir / "attack_vs_bitwise_distorted_mean.png",
            ylabel="bitwise/distorted (mean)",
        )

    if "hard/distorted_mean" in df_att.columns:
        plot_attack_metric_bars(
            df_att,
            metric="hard/distorted_mean",
            out_path=out_dir / "attack_vs_hard_distorted_mean.png",
            ylabel="hard/distorted (mean)",
        )

    # 2) Watermark / attack SNR footprint by attack
    if "sisnr_wm" in df_res.columns:
        plot_snr_box_by_attack(
            df_res,
            metric="sisnr_wm",
            out_path=out_dir / "sisnr_wm_box_by_attack.png",
        )
    if "sisnr_attack" in df_res.columns:
        plot_snr_box_by_attack(
            df_res,
            metric="sisnr_attack",
            out_path=out_dir / "sisnr_attack_box_by_attack.png",
        )

    # 3) Global feature importance
    plot_feature_importance_bar(
        df_fi,
        out_path=out_dir / "feature_importance_by_attack.png",
        top_k=args.top_k_features,
    )


if __name__ == "__main__":
    main()
