#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil

# 原始資料夾
SRC_ROOT = "/project/aimm/aki930/raw_bench/test_data"

# 要放「每個 dataset 取前 120 筆 .wav」的新資料夾
OUT_ROOT = "/project/aimm/aki930/raw_bench/test_data_10x120"

os.makedirs(OUT_ROOT, exist_ok=True)

for ds_name in sorted(os.listdir(SRC_ROOT)):
    ds_path = os.path.join(SRC_ROOT, ds_name)
    if not os.path.isdir(ds_path):
        continue

    # 找這個 dataset 裡的 .wav 檔
    # 如果 dataset 有次目錄，要遞迴就用 recursive=True 這一行；
    # 若檔案都平放在一層，可改成: glob.glob(os.path.join(ds_path, "*.wav"))
    wav_list = sorted(
        glob.glob(os.path.join(ds_path, "**", "*.wav"), recursive=True)
    )

    if not wav_list:
        print(f"[WARN] {ds_name} 裡沒有 .wav 檔，略過")
        continue

    selected = wav_list[:120]   # 只取前 120 個（不足 120 就全取）

    # 為這個 dataset 建一個對應的輸出資料夾
    out_ds_root = os.path.join(OUT_ROOT, ds_name)
    os.makedirs(out_ds_root, exist_ok=True)

    for src in selected:
        # 保留原本在 dataset 裡的相對路徑結構
        rel_path = os.path.relpath(src, ds_path)
        out_path = os.path.join(out_ds_root, rel_path)

        out_dir = os.path.dirname(out_path)
        os.makedirs(out_dir, exist_ok=True)

        shutil.copy2(src, out_path)

        # 如果你不在乎子資料夾結構，也可以改成：
        # out_path = os.path.join(out_ds_root, os.path.basename(src))
        # 然後直接 copy2(src, out_path)

# 全部 10 個 dataset 的「子集」都做好之後，把它們一起壓成 zip
zip_base_name = os.path.join(SRC_ROOT, "test_data_10x120")
zip_path = shutil.make_archive(
    base_name=zip_base_name,
    format="zip",
    root_dir=OUT_ROOT
)

print("Done! Zip 檔路徑：", zip_path)
