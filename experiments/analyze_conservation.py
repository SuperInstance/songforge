#!/usr/bin/env python3
"""Conservation-of-signal analysis for relay vs staircase rounds.

Measures, per file:
  - total energy (integral of power) — the conserved quantity if any
  - mean density (RMS) and density variance — staircase should be
    high-variance (choir body / silent tail), relay low-variance
  - voice-equivalents: mean power / mean single-voice power
  - silence fraction (windows below -45 dB) — where the holes are
  - opening/ending windows — does entry order still choose the ending?
Usage: python3 analyze_conservation.py <dir> <single-voice-dir>
"""
import os
import subprocess
import sys
import numpy as np


def read_wav(path):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True)
    data = np.frombuffer(out.stdout, dtype=np.float32)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
         "-of", "json", path], capture_output=True)
    import json
    info = json.loads(probe.stdout)
    sr = int(info["streams"][0]["sample_rate"])
    return data, sr


def analyze(path, single_power, winsize=0.5):
    data, sr = read_wav(path)
    dur = len(data) / sr
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    powers = (frames ** 2).mean(axis=1)  # per-window mean power
    total_energy = (data ** 2).sum() / sr
    mean_power = (data ** 2).mean()
    rms_db = 10 * np.log10(mean_power + 1e-12)
    pow_db = 10 * np.log10(powers + 1e-12)
    sil_frac = (pow_db < -45).mean()
    # body vs tail: last 15% of windows
    k = max(1, int(n * 0.15))
    body = pow_db[:-k].mean() if n > k else pow_db.mean()
    tail = pow_db[-k:].mean()
    # voice-equivalents (linear)
    veq = mean_power / single_power if single_power > 0 else float("nan")
    return {
        "file": os.path.basename(path), "dur": round(dur, 2),
        "total_energy": total_energy, "rms_db": round(float(rms_db), 2),
        "body_db": round(float(body), 2), "tail_db": round(float(tail), 2),
        "tail_minus_body": round(float(tail - body), 2),
        "silence_frac": round(float(sil_frac), 4),
        "voice_equiv": round(float(veq), 2),
        "profile_min": round(float(pow_db.min()), 1),
        "profile_std": round(float(pow_db.std()), 2),
    }


def main():
    d = sys.argv[1]
    single_dir = sys.argv[2]
    voices = ["lessac", "norman", "joe", "amy"]
    # single-voice mean power baseline
    spows = []
    for v in voices:
        data, sr = read_wav(os.path.join(single_dir, f"{v}.wav"))
        spows.append((data ** 2).mean())
    single_power = float(np.mean(spows))
    print(f"single-voice mean power baseline: {single_power:.6f} "
          f"({10*np.log10(single_power+1e-12):.1f} dB)")
    print("=" * 110)
    rows = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".wav"):
            continue
        p = os.path.join(d, f)
        rows.append(analyze(p, single_power))
    hdr = (f"{'file':<22}{'dur':>6}{'energy':>11}{'rms':>7}{'body':>7}"
           f"{'tail':>7}{'tail-body':>10}{'sil%':>7}{'veq':>6}{'std':>6}")
    print(hdr)
    for r in rows:
        print(f"{r['file']:<22}{r['dur']:>6}{r['total_energy']:>11.0f}"
              f"{r['rms_db']:>7}{r['body_db']:>7}{r['tail_db']:>7}"
              f"{r['tail_minus_body']:>10}{r['silence_frac']*100:>6.1f}%"
              f"{r['voice_equiv']:>6}{r['profile_std']:>6}")
    print("=" * 110)
    base = rows[0]["total_energy"] if rows else 1
    for r in rows:
        print(f"{r['file']:<22} energy/energy_staircase = "
              f"{r['total_energy']/base:.3f}")


if __name__ == "__main__":
    main()
