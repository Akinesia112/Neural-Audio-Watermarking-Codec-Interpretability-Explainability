#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_uncertainty_eval.py

從 test_results_*.csv 抽樣一些樣本，重新對「受攻擊後的音訊」做多次推論
(例如 TTA: 對輸入加小噪聲)，估計：

- mc_mean_prob : 多次預測的平均機率
- mc_var_prob  : 多次預測的機率變異數 (不確定性)
- attack_type  : 攻擊類型
- correct      : 以 mc_mean_prob 做 0.5 threshold 判斷，是否預測正確
- label        : 真實標籤（是否有水印）

結果存成一個 CSV，給 `analyze_uncertainty.py` 以及後續畫圖使用。
"""

import os
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from hydra import initialize, compose
from raw_bench.solver import SolverAudioSeal
from omegaconf import DictConfig

# ===== Monte Carlo / TTA 推論 =====

def monte_carlo_inference(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    n_iter: int = 20,
    noise_scale: float = 0.001,
) -> Tuple[float, float]:
    """
    對同一段 waveform 做多次推論，用「輸入加小噪聲」的方式估計不確定性。

    Args
    ----
    model : watermark detector (e.g. solver.model)
    waveform : [1, C, T] or [C, T] tensor on correct device
    n_iter : 總共 forward 次數
    noise_scale : 加到輸入上的高斯噪聲標準差

    Returns
    -------
    mean_prob : float
        多次預測的平均機率
    var_prob : float
        多次預測的機率變異數
    """
    model.eval()
    preds = []

    # 保證 shape 至少是 [1, C, T]
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        for _ in range(n_iter):
            if noise_scale > 0:
                noise = torch.randn_like(waveform) * noise_scale
                x = waveform + noise
            else:
                x = waveform

            # === 這一行是關鍵：如何從 model 得到「有水印」的機率 ===
            # 目前假設 model(x) 回傳一個 scalar logits/probability。
            # 如果你的模型介面不同，請在這裡改成正確的 forward & sigmoid。
            out = model(x)
            # 如果 out 不是 scalar，可以視情況 squeeze / 取某個維度
            prob = out.squeeze().item()
            preds.append(prob)

    preds = np.array(preds, dtype=np.float64)
    mean_prob = float(preds.mean())
    var_prob = float(preds.var())
    return mean_prob, var_prob


# ===== 主程式 =====

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="原本 eval 跑出來的 results CSV，例如 runs/.../results/test_results_loose.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="要輸出的 uncertainty CSV 路徑，例如 outputs/uncertainty/uncertainty_results_loose.csv",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="運算裝置，例如 'cuda' 或 'cpu'",
    )
    parser.add_argument(
        "--num_mc_samples",
        type=int,
        default=20,
        help="每個樣本要 forward 幾次做 Monte Carlo / TTA",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.001,
        help="輸入加噪聲的標準差（越大越抖動，越不穩定）",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="最多抽多少筆樣本來做不確定性分析（避免整個 test 全部重跑太慢）",
    )
    parser.add_argument(
        "--csv_sep",
        type=str,
        default="|",
        help="results_csv 的欄位分隔符號，預設是 '|'，如果你的 CSV 是逗號，就設 ','",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="audioseal/eval_loose",
        help=(
            "Hydra config 名稱（不含 .yaml），預設 audioseal/eval_loose。\n"
            "會從 ../configs 底下讀，例如 ../configs/audioseal/eval_loose.yaml"
        ),
    )
    return parser.parse_args()



def load_eval_config(config_name: str) -> DictConfig:
    """
    使用 Hydra 從 configs/ 底下載入 eval 設定。
    假設執行指令是在專案根目錄 raw_bench/ 下：
        python scripts/run_uncertainty_eval.py ...

    那麼 config_path 就是相對路徑 "configs"，例如：
        configs/audioseal/eval_loose.yaml
    """
    from hydra import initialize, compose

    # 一定要是 *相對路徑*，不能是 /project/... 這種絕對路徑
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name=config_name)  # e.g. "audioseal/eval_loose"

    return cfg


def main():
    args = parse_args()

    results_path = Path(args.results_csv)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. 讀取原本的 results CSV
    print(f"[INFO] Loading results table from {results_path}")
    df = pd.read_csv(results_path, sep=args.csv_sep)

    # 這裡只做 subset，避免太慢
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"[INFO] Subsampled to {len(df)} rows for uncertainty evaluation.")
    else:
        print(f"[INFO] Using all {len(df)} rows for uncertainty evaluation.")

    # 2. 用 Hydra 當 library 讀 eval config，然後初始化 SolverAudioSeal
    print(f"[INFO] Loading eval config '{args.config_name}' via Hydra...")
    cfg = load_eval_config(args.config_name)
    # 如果 config 裡有 device 欄位，可以順便覆蓋
    try:
        cfg.device = args.device
    except Exception:
        pass

    print("[INFO] Initializing SolverAudioSeal...")
    solver = SolverAudioSeal(cfg)
    # 這一行假設 detector 存在 solver.model，你可以依照實際 class 改成 solver.detector 等
    model = solver.model.to(args.device)
    model.eval()

    # 3. 對每一筆樣本做 Monte Carlo 評估
    records = []

    print("[INFO] Running Monte Carlo / uncertainty evaluation...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # === 這裡要從 results_csv 裡拿到「受攻擊後的音檔路徑」 ===
        # 下面假設欄位名稱是 audio_filepath；如果你實際欄位不同，請改這一行。
        if "audio_filepath" in row:
            wav_path = row["audio_filepath"]
        elif "wav_path" in row:
            wav_path = row["wav_path"]
        else:
            # 如果沒有相關欄位，就跳過（你可以改成 raise error）
            continue

        wav_path = Path(wav_path)
        if not wav_path.is_file():
            # 如果 CSV 裡存的是相對路徑，你可以在這裡加上根目錄
            # 例如 root = results_path.parent.parent / "data"
            # wav_path = root / wav_path
            # 這邊我先簡單跳過缺失檔案
            continue

        try:
            wav, sr = torchaudio.load(str(wav_path))
        except Exception:
            continue

        # 保證 waveform 是 mono，shape 至少 [1, T]
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(args.device)

        # 取 label（是否有水印）
        label = None
        if "is_watermarked" in df.columns:
            label = int(row["is_watermarked"])
        elif "label" in df.columns:
            label = int(row["label"])
        # 否則 label 可能為 None；這樣 correct 會設成 None

        # 做 Monte Carlo / TTA
        mean_prob, var_prob = monte_carlo_inference(
            model=model,
            waveform=wav,
            n_iter=args.num_mc_samples,
            noise_scale=args.noise_scale,
        )

        rec = {
            "audio_filepath": str(wav_path),
            "attack_type": row.get("attack_type", None),
            "dataset": row.get("dataset", None),
            "chunk_index": row.get("chunk_index", None),
            "mc_mean_prob": mean_prob,
            "mc_var_prob": var_prob,
        }

        if label is not None:
            rec["label"] = label
            rec["correct"] = int((mean_prob >= 0.5) == bool(label))
        else:
            rec["label"] = None
            rec["correct"] = None

        records.append(rec)

    out_df = pd.DataFrame(records)
    out_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved uncertainty table to {out_path}")


if __name__ == "__main__":
    main()
