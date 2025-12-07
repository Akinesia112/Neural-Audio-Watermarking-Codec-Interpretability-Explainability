#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def load_table(path: Path) -> pd.DataFrame:
    with path.open("r") as f:
        header = f.readline()
    sep = "|" if "|" in header else ","
    return pd.read_csv(path, sep=sep)


def reliability_curve(probs, correct, n_bins=10):
    """回傳 bin_center, bin_acc, bin_conf"""
    probs = np.asarray(probs)
    correct = np.asarray(correct).astype(bool)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    acc = np.zeros(n_bins)
    conf = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            acc[b] = np.nan
            conf[b] = np.nan
        else:
            acc[b] = correct[mask].mean()
            conf[b] = probs[mask].mean()
    return bin_centers, acc, conf


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze uncertainty CSV from run_uncertainty_eval.py\n\n"
            "需要 CSV 裡有欄位：\n"
            "  mc_mean_prob, mc_var_prob, attack_var_prob (optional),\n"
            "  attack_type (optional),\n"
            "  hard/distorted 或 correct (用來判斷有沒有偵測對)。"
        )
    )
    parser.add_argument(
        "--uncertainty_csv",
        type=str,
        required=True,
        help="CSV produced by run_uncertainty_eval.py",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory for plots.",
    )
    parser.add_argument(
        "--correct_col",
        type=str,
        default="hard/distorted",
        help="欄位名，用來判斷是否偵測正確；>0.5 視為正確。",
    )
    parser.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="Reliability curve 的 bin 數。",
    )
    args = parser.parse_args()

    path = Path(args.uncertainty_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_table(path)

    # 定義 correct label
    if "correct" in df.columns:
        correct = df["correct"].astype(bool).values
    else:
        # 預設用 hard/distorted > 0.5
        if args.correct_col not in df.columns:
            raise ValueError(
                f"Could not find 'correct' or '{args.correct_col}' in CSV."
            )
        correct = (df[args.correct_col].values > 0.5)

    # ---- 1) mc_var_prob / attack_var_prob 分佈：correct vs wrong ----
    for var_col in ["mc_var_prob", "attack_var_prob"]:
        if var_col not in df.columns:
            continue

        v = df[var_col].values
        v_correct = v[correct]
        v_wrong = v[~correct]

        plt.figure(figsize=(6, 4))
        plt.hist(v_correct, bins=50, alpha=0.6, label="correct", density=True)
        plt.hist(v_wrong, bins=50, alpha=0.6, label="wrong", density=True)
        plt.xlabel(var_col)
        plt.ylabel("density")
        plt.title(f"{var_col}: correct vs wrong")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{var_col}_correct_vs_wrong_hist.png")
        plt.close()
        print(f"  -> {var_col}_correct_vs_wrong_hist.png")

        # ROC: 用 variance 預測「會不會錯」
        #   y_true = 1 代表「錯」（bad prediction）
        y_true = (~correct).astype(int)
        # 越大 variance 越不確定 → 越像 "wrong"
        fpr, tpr, _ = roc_curve(y_true, v)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC using {var_col} as uncertainty score")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"{var_col}_roc.png")
        plt.close()
        print(f"  -> {var_col}_roc.png (AUC={roc_auc:.3f})")

    # ---- 2) reliability curve: mc_mean_prob 當 confidence ----
    if "mc_mean_prob" in df.columns:
        probs = df["mc_mean_prob"].values
        centers, acc, conf = reliability_curve(
            probs, correct, n_bins=args.n_bins
        )

        plt.figure(figsize=(4, 4))
        mask = ~np.isnan(acc)
        plt.plot(conf[mask], acc[mask], "o-")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("predicted probability (bin mean)")
        plt.ylabel("empirical accuracy")
        plt.title("Reliability curve (MC mean prob)")
        plt.tight_layout()
        plt.savefig(out_dir / "reliability_curve_mc_mean_prob.png")
        plt.close()
        print("  -> reliability_curve_mc_mean_prob.png")

    # ---- 3) optional: per-attack breakdown of variance ----
    if "attack_type" in df.columns and "mc_var_prob" in df.columns:
        attacks = sorted(df["attack_type"].unique())
        data = [df.loc[df["attack_type"] == a, "mc_var_prob"].values for a in attacks]

        plt.figure(figsize=(10, 4))
        plt.boxplot(data, labels=attacks, showfliers=False)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mc_var_prob")
        plt.title("MC variance by attack_type")
        plt.tight_layout()
        plt.savefig(out_dir / "mc_var_prob_box_by_attack.png")
        plt.close()
        print("  -> mc_var_prob_box_by_attack.png")

    print("[analyze_uncertainty] Done.")


if __name__ == "__main__":
    main()
