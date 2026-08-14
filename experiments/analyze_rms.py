#!/usr/bin/env python3
"""Analyze RMS levels and spectral bands of wav files.
Usage: python3 analyze_rms.py <file.wav> [--tail N] [--window S] [--bands]
Prints full-file RMS, peak, and (optionally) per-band RMS in dB.
Also can print a temporal density profile (RMS per second) with --profile.
"""
import sys
import subprocess
import json
import numpy as np


def read_wav(path):
    """Read wav via ffmpeg into float32 mono numpy array."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True,
    )
    data = np.frombuffer(out.stdout, dtype=np.float32)
    sr = None
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate",
         "-of", "json", path],
        capture_output=True,
    )
    info = json.loads(probe.stdout)
    sr = int(info["streams"][0]["sample_rate"])
    return data, sr


def rms_db(x):
    if len(x) == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(x)))
    if rms < 1e-9:
        return -120.0
    return 20 * np.log10(rms)


def peak_db(x):
    peak = np.max(np.abs(x)) if len(x) else 0
    if peak < 1e-9:
        return -120.0
    return 20 * np.log10(peak)


def band_rms(x, sr, lo, hi):
    """Simple band split via FFT filtering (rectangular in freq domain)."""
    n = len(x)
    if n < 1024:
        return -120.0
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    mask = (freqs >= lo) & (freqs < hi)
    Xb = X * mask
    y = np.fft.irfft(Xb, n)
    return rms_db(y)


def main():
    args = sys.argv[1:]
    path = args[0]
    do_profile = "--profile" in args
    tail = None
    if "--tail" in args:
        tail = float(args[args.index("--tail") + 1])

    x, sr = read_wav(path)
    dur = len(x) / sr
    full = rms_db(x)
    peak = peak_db(x)
    print(f"{path}")
    print(f"  duration: {dur:.3f}s  sr: {sr}")
    print(f"  full RMS: {full:.1f} dB   peak: {peak:.1f} dB")

    if tail is not None:
        tx = x[int((dur - tail) * sr):]
        print(f"  tail ({tail}s) RMS: {rms_db(tx):.1f} dB")
        print(f"    tail low  (20-250Hz):  {band_rms(tx, sr, 20, 250):.1f} dB")
        print(f"    tail mid  (250-4000Hz): {band_rms(tx, sr, 250, 4000):.1f} dB")
        print(f"    tail high (4k-12k):    {band_rms(tx, sr, 4000, 12000):.1f} dB")

    if do_profile:
        win = 0.5
        step = 0.25
        ts = np.arange(0, dur - win, step)
        vals = []
        for t in ts:
            seg = x[int(t * sr):int((t + win) * sr)]
            vals.append(rms_db(seg))
        print(f"  profile ({win}s window, {step}s step):")
        print("  " + " ".join(f"{t:.2f}:{v:.1f}" for t, v in zip(ts, vals)))


if __name__ == "__main__":
    main()
