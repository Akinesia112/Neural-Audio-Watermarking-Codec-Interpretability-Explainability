import os
import csv

MODEL = "SemanticRandom"

ROOT = f"../watermark_research/results/LibriSpeech-init/{MODEL}/"
OUT_CSV = f"{MODEL}_pairs.csv"

rows = []

for folder in sorted(os.listdir(ROOT)):
    sub = os.path.join(ROOT, folder)
    if not os.path.isdir(sub):
        continue

    clean = os.path.join(sub, "1_original.wav")
    wm = os.path.join(sub, "2_watermarked.wav")

    if not (os.path.exists(clean) and os.path.exists(wm)):
        print("Skipping (missing files):", folder)
        continue

    rows.append({
        "clean": clean,
        "watermarked": wm,
        "wm_method": MODEL,        # change manually
        "instrument": "speech"
    })

print("Found", len(rows), "pairs.")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["clean", "watermarked", "wm_method", "instrument"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print("Wrote:", OUT_CSV)
