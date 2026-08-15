#!/usr/bin/env python3
"""Session 69, experiment 1: THE SATURATION POINT — container-tax curve past N=8.

S68 found the medium's ceiling curve: container tax (s16 energy / f32 energy)
0.9976 (N=2) -> 0.9920 (N=4) -> 0.9738 (N=8), monotone in N — the s16
container levies a larger share as the census grows.  Open questions:

  A. Where does the curve GO?  Does the tax keep falling smoothly, or does
     it break at some N where the clipped fraction suddenly dominates?
  B. Is there a saturation N where the s16 census records as FLAT-TOPPED
     SILENCE — a crowd large enough that the container can no longer hold
     the sum, and the recording is a wall of ±32767?
  C. The census law in the f32 medium: veq(N) = N + bonus(N).  S68 bonus:
     0.010 (N=2) -> 0.025 (N=4) -> 0.066 (N=8) — superlinear in N.  What
     is the true exponent?  (bonus ~ N^1.36 from three points)

Measures, per N in {1, 2, 4, 8, 16, 32}:
  - veq(s16), veq(f32)            the census in each container
  - container tax                 s16 energy / f32 energy
  - clip fraction                 fraction of pre-clip samples |mix| >= 1.0
  - flat-top fraction             fraction of CLIPPED samples pinned at ±1.0
  - bonus exponent fit            veq_f32(N) = N + a*N^b

Usage: python3 session69_taxcurve.py <session66_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session68_census import read_f32, write_wav
import analyze_conservation as AC

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")


def staircase_pair(voices, out_s16, out_f32, interval, sr):
    """Same census in s16 and f32; returns (mix, clip_frac, flat_frac)."""
    datas = [read_f32(v) for v in voices]
    end = max(int(len(d) / sr + i * interval) for i, d in enumerate(datas))
    n = (end + 1) * sr
    mix = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(datas):
        s = int(i * interval * sr)
        mix[s:s + len(d)] += d
    clip_frac = float((np.abs(mix) >= 1.0).mean())
    write_wav(mix, sr, out_s16, "s16le")
    write_wav(mix, sr, out_f32, "f32le")
    # flat-top: samples that were clipped land exactly on the rails
    s16 = np.clip(mix, -1.0, 1.0)
    flat_frac = float((np.abs(s16) >= 1.0 - 1e-9).mean())
    return mix, clip_frac, flat_frac


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session69/taxcurve"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())

    # reference single-voice TOTAL ENERGY (S68 convention: veq = energy
    # ratio; mean-power ratios are diluted by the +1 s zero pad of the
    # staircase builder and by the entry ramp)
    ref_energy = AC.analyze(cast[1], 1.0)["total_energy"]

    print("=" * 72)
    print("THE SATURATION POINT (container-tax curve, N = 1..32)")
    print("=" * 72)
    rows = {}
    for N in (1, 2, 4, 8, 16, 32):
        voices = (cast * (N // len(cast) + 1))[:N]
        base = os.path.join(out, f"tax-N{N:02d}")
        _, clip_frac, flat_frac = staircase_pair(
            voices, base + "-s16.wav", base + "-f32.wav", 1.3, sr)
        e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
        p16 = AC.analyze(base + "-s16.wav", 1.0)["rms_db"]
        tax = e16 / e32
        veq16 = e16 / ref_energy
        veq32 = e32 / ref_energy
        rows[N] = {"veq_s16": veq16, "veq_f32": veq32,
                   "tax": float(tax), "clip_frac": clip_frac,
                   "flat_frac": flat_frac, "rms_db_s16": p16}
        print(f"N={N:2d}: veq(s16) {veq16:7.3f}  veq(f32) {veq32:7.3f}  "
              f"tax {tax:.4f}  clip {clip_frac*100:5.2f}%  "
              f"flat {flat_frac*100:5.2f}%")

    # ---- the census dividend: veq_f32(N) = N + a*N^b
    print()
    print("=" * 72)
    print("THE CENSUS DIVIDEND (bonus = veq_f32 - N, fit a*N^b)")
    print("=" * 72)
    ns = np.array(sorted(rows))
    bonus = {n: rows[n]["veq_f32"] - n for n in ns}
    # fit on the positive-dividend points only (N=2,4,8 match S68; the
    # dividend flips sign at N=16/32 when more twin lags sum to zero)
    fit_n = np.array([n for n in ns if bonus[n] > 0])
    fit_b = np.array([bonus[n] for n in fit_n])
    logb, logn = np.log(fit_b), np.log(fit_n)
    b_fit, a_fit = np.polyfit(logn, logb, 1)
    pred = np.exp(a_fit) * ns ** b_fit
    for n, bo, p in zip(ns, [bonus[n] for n in ns], pred):
        print(f"N={n:2d}: bonus {bo:+.4f}   fit a*N^b "
              f"{p:+.4f}  (a={np.exp(a_fit):.4f}, b={b_fit:.3f})")
    rows["_fit"] = {"a": float(np.exp(a_fit)), "b": float(b_fit),
                    "series_s68": [0.010, 0.025, 0.066],
                    "series_this_run": [float(bonus[n]) for n in ns]}    # ---- where does s16 saturate?  flat-top fraction crossing points
    print()
    print("=" * 72)
    print("THE FLAT-TOPPED SILENCE THRESHOLD")
    print("=" * 72)
    for N in (8, 16, 32):
        r = rows[N]
        print(f"N={N:2d}: {r['flat_frac']*100:.2f}% of the census is "
              f"pinned at the rails (flat-topped); {r['clip_frac']*100:.2f}% "
              f"of samples were clipped before the container could write them")

    with open(os.path.join(out, "taxcurve-report.json"), "w") as fj:
        json.dump(rows, fj, indent=1, default=str)
    print(f"\nreport -> {os.path.join(out, 'taxcurve-report.json')}")


if __name__ == "__main__":
    main()
