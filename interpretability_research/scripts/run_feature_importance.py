#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_table(path: Path) -> pd.DataFrame:
    """auto detect ',' / '|' delimiter."""
    with path.open("r") as f:
        header = f.readline()
    sep = "|" if "|" in header else ","
    return pd.read_csv(path, sep=sep)


def compute_feature_importance(
    df: pd.DataFrame, target: str, exclude_cols=None
) -> pd.DataFrame:
    """
    給一個聚合好的 stats 表，對 target 做簡單 correlation-based feature importance。

    - 只用 numeric 欄位
    - 排除 target 本身和 exclude_cols
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in dataframe.")

    if exclude_cols is None:
        exclude_cols = []

    y = df[target].astype(float)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c != target and c not in exclude_cols]

    rows = []
    for feat in features:
        x = df[feat].astype(float)
        if x.nunique(dropna=True) <= 1:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = x.corr(y, method="pearson")
            spearman = x.corr(y, method="spearman")

        rows.append(
            dict(
                feature=feat,
                pearson=pearson,
                abs_pearson=abs(pearson) if pd.notna(pearson) else np.nan,
                spearman=spearman,
                abs_spearman=abs(spearman) if pd.notna(spearman) else np.nan,
            )
        )

    imp = pd.DataFrame(rows).sort_values("abs_pearson", ascending=False)
    return imp


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Feature-importance on aggregated watermark stats.\n\n"
            "Example:\n"
            "  python scripts/run_feature_importance.py \\\n"
            "    --stats_csv outputs/wm_stats_loose/stats_by_attack.csv \\\n"
            "    --target_metric bitwise/distorted_mean \\\n"
            "    --out_csv outputs/wm_stats_loose/feature_importance_by_attack.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stats_csv",
        type=str,
        required=True,
        help="stats_by_attack.csv / stats_by_dataset_attack.csv / ...",
    )
    parser.add_argument(
        "--target_metric",
        type=str,
        required=True,
        help="要當作 y 的欄位，例如 'bitwise/distorted_mean'。",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="輸出的 feature importance CSV。",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="不當作 feature 的欄位（例如 'n' 或某些 helper 指標）。",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="在 stdout 印出的前 K 名 features。",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = load_table(stats_path)
    imp = compute_feature_importance(
        df, target=args.target_metric, exclude_cols=args.exclude
    )

    imp.to_csv(out_path, index=False)
    print(f"[run_feature_importance] Saved to {out_path}")

    print("\nTop features (by |pearson|):")
    print(imp.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
