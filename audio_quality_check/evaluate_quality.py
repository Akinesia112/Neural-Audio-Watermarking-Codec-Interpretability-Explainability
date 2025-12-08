import os
import glob
import numpy as np
import soundfile as sf
import librosa
from scipy import stats
import math
import csv

# Optional libs (pip install pesq pystoi mir_eval)
try:
    from pesq import pesq
except Exception:
    pesq = None
try:
    from pystoi import stoi
except Exception:
    stoi = None

def load_audio(path, sr=16000, mono=True):
    y, fs = sf.read(path)
    if mono and y.ndim==2:
        y = np.mean(y, axis=1)
    if fs != sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=fs, target_sr=sr)
        fs = sr
    return y, fs

def si_snr(est, ref, eps=1e-8):
    # scale-invariant SNR, from literature
    # est, ref are 1d numpy
    assert est.shape == ref.shape
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    s_target = np.sum(ref * est) * ref / (np.sum(ref**2) + eps)
    e_noise = est - s_target
    return 10 * np.log10((np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps))

def simple_snr(ref, est, eps=1e-8):
    # classic SNR = 10*log10( signal_power / noise_power )
    sig = np.sum(ref**2)
    noise = np.sum((ref - est)**2)
    return 10*np.log10((sig + eps) / (noise + eps))

def log_spectral_distance(ref, est, n_fft=1024, hop=512, eps=1e-8):
    # compute log-spectral distance averaged over frames
    ref_spec = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop)) + eps
    est_spec = np.abs(librosa.stft(est, n_fft=n_fft, hop_length=hop)) + eps
    ref_db = 20*np.log10(ref_spec)
    est_db = 20*np.log10(est_spec)
    lsd = np.sqrt(np.mean((ref_db - est_db)**2, axis=0))  # per frame
    return float(np.mean(lsd))

def compute_metrics(clean_path, wm_path, sr=16000):
    clean, fs = load_audio(clean_path, sr=sr)
    wm, _ = load_audio(wm_path, sr=sr)
    # align lengths
    L = min(len(clean), len(wm))
    clean = clean[:L]
    wm = wm[:L]
    metrics = {}
    metrics['si_snr_clean'] = si_snr(clean, clean)  # trivially inf/large, keep as baseline
    metrics['si_snr_watermarked'] = si_snr(wm, clean)
    metrics['delta_si_snr'] = metrics['si_snr_watermarked'] - metrics['si_snr_clean']
    metrics['snr'] = simple_snr(clean, wm)
    metrics['lsd'] = log_spectral_distance(clean, wm)
    if pesq is not None:
        try:
            # pesq requires fs 16000 or 8000 for narrowband, some versions support wideband
            metrics['pesq'] = pesq(fs, clean, wm, 'wb')
        except Exception:
            metrics['pesq'] = None
    if stoi is not None:
        try:
            metrics['stoi'] = stoi(clean, wm, fs, extended=False)
        except Exception:
            metrics['stoi'] = None
    return metrics

def evaluate_pairs(pairs_csv, out_csv='quality_results.csv', sr=16000):
    """
    pairs_csv format: clean_path, watermarked_path, wm_method, instrument, (optional metadata)
    """
    results = []
    with open(pairs_csv, 'r', newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            print("READ ROW:", row)
            clean_path = row['clean']
            wm_path = row['watermarked']
            if not os.path.exists(clean_path):
                print("CLEAN NOT FOUND:", clean_path)
                continue
            if not os.path.exists(wm_path):
                print("WM NOT FOUND:", wm_path)
                continue
            wm_method = row.get('wm_method','')
            instrument = row.get('instrument','')
            try:
                m = compute_metrics(clean_path, wm_path, sr=sr)
            except Exception as e:
                print("METRIC ERROR for:", clean_path, wm_path)
                print("Error: ", e)
                continue
            out = {'clean': clean_path, 'watermarked': wm_path, 'wm_method': wm_method, 'instrument':instrument}
            out.update(m)
            results.append(out)
    # save
    keys = list(results[0].keys())
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    return results

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', required=True, help='CSV with columns: clean,watermarked,wm_method,instrument')
    p.add_argument('--out', default='quality_results.csv')
    p.add_argument('--sr', type=int, default=16000)
    args = p.parse_args()
    res = evaluate_pairs(args.pairs, out_csv=args.out, sr=args.sr)
    print("Done. Wrote", args.out)
