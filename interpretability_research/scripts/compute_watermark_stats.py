#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def agg_stats(df: pd.DataFrame, group_cols):
    """
    對所有 numeric 欄位做 mean / std 聚合。

    group_cols 為空時，回傳一列 overall stats（不再丟 "No group keys"）。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not group_cols:
        out = {"n": len(df)}
        for c in numeric_cols:
            out[c + "_mean"] = df[c].mean()
            out[c + "_std"] = df[c].std(ddof=0)
        return pd.DataFrame([out])

    grouped = df.groupby(group_cols, dropna=False)

    rows = []
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n"] = len(sub)
        for c in numeric_cols:
            row[c + "_mean"] = sub[c].mean()
            row[c + "_std"] = sub[c].std(ddof=0)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compute aggregate watermark stats from raw_bench test_results_*.csv."
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="Path to test_results_loose.csv (pipe '|' delimited).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to write aggregated CSVs.",
    )
    parser.add_argument(
        "--by_chunk_index",
        action="store_true",
        help="Also aggregate stats over chunk_index / (attack_type, chunk_index) if present.",
    )

    args = parser.parse_args()

    results_path = Path(args.results_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compute_watermark_stats] Loading {results_path}")
    df = pd.read_csv(results_path, sep="|")

    # 1) overall
    overall = agg_stats(df, [])
    overall.to_csv(out_dir / "overall_stats.csv", index=False)
    print(f"  -> overall_stats.csv ({len(overall)} row)")

    # 2) by attack
    if "attack_type" in df.columns:
        by_attack = agg_stats(df, ["attack_type"])
        by_attack.to_csv(out_dir / "stats_by_attack.csv", index=False)
        print(f"  -> stats_by_attack.csv ({len(by_attack)} rows)")

    # 3) by dataset
    if "dataset" in df.columns:
        by_dataset = agg_stats(df, ["dataset"])
        by_dataset.to_csv(out_dir / "stats_by_dataset.csv", index=False)
        print(f"  -> stats_by_dataset.csv ({len(by_dataset)} rows)")

    # 4) by dataset + attack
    if all(c in df.columns for c in ["dataset", "attack_type"]):
        by_ds_att = agg_stats(df, ["dataset", "attack_type"])
        by_ds_att.to_csv(out_dir / "stats_by_dataset_attack.csv", index=False)
        print(f"  -> stats_by_dataset_attack.csv ({len(by_ds_att)} rows)")

    # 5) optional: by chunk_index
    if args.by_chunk_index and "chunk_index" in df.columns:
        by_chunk = agg_stats(df, ["chunk_index"])
        by_chunk.to_csv(out_dir / "stats_by_chunk_index.csv", index=False)
        print(f"  -> stats_by_chunk_index.csv ({len(by_chunk)} rows)")

        if all(c in df.columns for c in ["attack_type", "chunk_index"]):
            by_att_chunk = agg_stats(df, ["attack_type", "chunk_index"])
            by_att_chunk.to_csv(
                out_dir / "stats_by_attack_chunk_index.csv", index=False
            )
            print(
                f"  -> stats_by_attack_chunk_index.csv ({len(by_att_chunk)} rows)"
            )

    print("[compute_watermark_stats] Done.")


if __name__ == "__main__":
    main()
