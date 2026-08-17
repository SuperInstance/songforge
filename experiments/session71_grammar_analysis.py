#!/usr/bin/env python3
"""SongForge Session 71 — Grammar experiment analysis.

Blind-ish scoring harness for the A/B/C prompt-grammar experiment.
Measures objective audio features per track:
  - duration, RMS/peak loudness, dynamic range
  - spectral centroid mean + evolution (change over time = "evolution")
  - spectral flatness (timbral character)
  - zero-crossing rate (brightness proxy)
  - band-energy ratios (low/mid/high) for genre fingerprinting
  - onset density (rhythmic activity proxy)
  - tail behavior (does it end with a sustained tone / solitary note?)

Writes a JSON report + a markdown table suitable for blind scoring.
"""
import sys, os, json, glob
import numpy as np
import soundfile as sf

def analyze(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    dur = n / sr

    # loudness
    rms = np.sqrt(np.mean(data ** 2))
    peak = np.max(np.abs(data))
    # dynamic range: 10th vs 90th percentile of frame RMS
    fr = 2048
    hop = 512
    nf = max(1, (n - fr) // hop)
    frames = np.array([data[i*hop:i*hop+fr] for i in range(nf)])
    frms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    dr_db = 20 * np.log10(np.percentile(frms, 90) / max(np.percentile(frms, 10), 1e-12))

    # spectral features per frame
    win = np.hanning(fr)
    spec = np.abs(np.fft.rfft(frames * win, axis=1))
    freqs = np.fft.rfftfreq(fr, 1 / sr)
    ps = spec ** 2
    psum = ps.sum(axis=1) + 1e-12
    centroid = (ps * freqs).sum(axis=1) / psum
    # flatness (geometric / arithmetic mean of power)
    flat = np.exp(np.mean(np.log(ps + 1e-12), axis=1)) / (np.mean(ps, axis=1) + 1e-12)
    # band energy
    lo = (freqs < 250).sum()
    mid = ((freqs >= 250) & (freqs < 2000)).sum()
    hi = (freqs >= 2000).sum()
    band_lo = ps[:, :lo].sum(axis=1) / psum
    band_mid = ps[:, lo:lo+mid].sum(axis=1) / psum
    band_hi = ps[:, lo+mid:lo+mid+hi].sum(axis=1) / psum

    # zero-crossing rate
    zcr = np.mean(np.abs(np.diff(np.signbit(frames).astype(int), axis=1)), axis=1)

    # onset density: frames where RMS jumps > 2x previous (smoothed)
    env = np.convolve(frms, np.ones(5) / 5, mode="same")
    onset = np.sum(np.diff(env > 0.02 * np.percentile(env, 95)) > 0) / dur

    # evolution: std of smoothed centroid over time / mean (how much it moves)
    c_smooth = np.convolve(centroid, np.ones(20) / 20, mode="same")
    evolution = np.std(c_smooth) / (np.mean(centroid) + 1e-9)

    # tail: last 8 s — mean RMS vs body RMS (a sustained-tone ending = tail RMS high, low variance)
    tail_s = 8
    tail_n = int(tail_s * sr)
    tail_rms = np.sqrt(np.mean(data[-tail_n:] ** 2)) if n > tail_n else rms
    tail_var = np.var(data[-tail_n:]) if n > tail_n else 0.0

    return {
        "file": os.path.basename(path),
        "duration_s": round(dur, 2),
        "rms_db": round(20 * np.log10(rms + 1e-12), 2),
        "peak_db": round(20 * np.log10(peak + 1e-12), 2),
        "dynamic_range_db": round(dr_db, 2),
        "centroid_mean_hz": round(float(np.mean(centroid)), 1),
        "centroid_std_hz": round(float(np.std(centroid)), 1),
        "evolution_score": round(float(evolution), 4),
        "flatness_mean": round(float(np.mean(flat)), 4),
        "band_lo_frac": round(float(np.mean(band_lo)), 3),
        "band_mid_frac": round(float(np.mean(band_mid)), 3),
        "band_hi_frac": round(float(np.mean(band_hi)), 3),
        "zcr_mean": round(float(np.mean(zcr)), 4),
        "onset_density": round(float(onset), 2),
        "tail_rms_db": round(20 * np.log10(tail_rms + 1e-12), 2),
        "tail_var": round(float(tail_var), 6),
    }

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "audio/session71/grammar"
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.mp3"))):
        try:
            out.append(analyze(p))
        except Exception as e:
            print(f"ERR {p}: {e}", file=sys.stderr)
    out.sort(key=lambda r: r["file"])
    report = os.path.join(d, "grammar-analysis.json")
    with open(report, "w") as f:
        json.dump(out, f, indent=1)

    # markdown table
    cols = ["file", "duration_s", "rms_db", "dynamic_range_db", "centroid_mean_hz",
            "evolution_score", "flatness_mean", "band_lo_frac", "band_mid_frac",
            "band_hi_frac", "onset_density", "tail_rms_db"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in out:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")
    print(f"\nWrote {report} ({len(out)} tracks)")

if __name__ == "__main__":
    main()
