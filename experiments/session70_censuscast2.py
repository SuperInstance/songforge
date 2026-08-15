#!/usr/bin/env python3
"""Session 70, experiment 2b: THE CENSUS WITH A DIFFERENT CAST (fixed).

2a's negative dividend was a LOUDNESS ARTIFACT: veq = total/ref_energy
with ref = lessac only; norman/joe/amy are quieter, so total < N*ref
even with zero correlation.  Fix: normalize every voice to EQUAL total
energy before the staircase, so bonus = veq_f32 - N is pure correlation
surplus/deficit — the honest census question.

This makes the cast comparison fair:
  - roror cast (S69): 4 near-identical renditions of one phrase
  - s64 cast: 4 distinct speakers, distinct utterances
Question: does the dividend sign-flip at N=16/32 (roror: +0.066 @ 8,
~0 @ 16, -0.403 @ 32) hold for genuinely different material, or was it
the identical-cast twin effect (the crowd undercounts itself only when
the crowd is one voice many times)?

Secondary measure: the overlap duty cycle — fraction of mix time where
>= 2 voices are present.  Short s64 voices at 1.3 s spacing barely
overlap (duty ~ (len - interval)/len per pair); the roror cast at 240 s
voices overlaps almost everywhere.  The tax curve difference
(0.9924 @ 32 for s64 vs 0.7864 for roror) may be pure overlap, not
cast identity.  Report both.

Usage: python3 session70_censuscast2.py <session64_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session68_census import read_f32, write_wav
import analyze_conservation as AC

S64 = ("lessac.wav", "norman.wav", "joe.wav", "amy.wav")
NS = (1, 2, 4, 8, 16, 32)
INTERVAL = 1.3
SR = 22050.0


def normalize(voices):
    """Return energy-normalized copies (each total energy == 1.0)."""
    out = []
    for v in voices:
        d = read_f32(v)
        e = float((d ** 2).sum())
        out.append(d * np.sqrt(1.0 / e) if e > 0 else d)
    return out


def staircase_norm(datas, out_s16, out_f32, interval, sr):
    """Census on pre-normalized datas; also returns overlap duty."""
    end = max(int(len(d) / sr + i * interval) for i, d in enumerate(datas))
    n = int((end + 1) * sr)
    mix = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(datas):
        s = int(i * interval * sr)
        mix[s:s + len(d)] += d
    # overlap duty: time with >= 2 voices / total time
    on = np.zeros(n, dtype=np.int8)
    for i, d in enumerate(datas):
        s = int(i * interval * sr)
        on[s:s + len(d)] += 1
    duty = float((on >= 2).mean())
    clip_frac = float((np.abs(mix) >= 1.0).mean())
    write_wav(mix, sr, out_s16, "s16le")
    write_wav(mix, sr, out_f32, "f32le")
    return clip_frac, duty


def main():
    s64 = sys.argv[1] if len(sys.argv) > 1 else "audio/session64"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/censuscast2"
    os.makedirs(out, exist_ok=True)

    cast = [os.path.join(s64, n) for n in S64]
    datas = normalize(cast)          # each voice total energy == 1.0
    ref_energy = 1.0                 # every voice is the ref now
    # overlap duty of the full N=32 build (representative)
    _, duty32 = staircase_norm(datas * 8, os.path.join(out, "duty-N32.wav"),
                               os.path.join(out, "duty-N32-f32.wav"),
                               INTERVAL, SR)

    print("=" * 72)
    print("THE CENSUS WITH A DIFFERENT CAST (s64, ENERGY-NORMALIZED)")
    print("=" * 72)
    rows = {}
    for N in NS:
        ds = (datas * (N // len(datas) + 1))[:N]
        base = os.path.join(out, f"cast2-N{N:02d}")
        clip_frac, duty = staircase_norm(ds, base + "-s16.wav",
                                         base + "-f32.wav", INTERVAL, SR)
        e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
        # normalized voices have sum(x^2) == 1.0, so the true census is
        # total_energy * sr (AC divides by sr).
        veq32 = e32 * SR
        rows[N] = {"veq_f32": veq32, "tax": float(e16 / e32),
                   "clip_frac": clip_frac, "duty": duty}
        print(f"N={N:2d}: veq(f32) {veq32:7.3f}  "
              f"bonus {veq32 - N:+.4f}  tax {e16/e32:.4f}  "
              f"clip {clip_frac*100:5.2f}%  duty {duty*100:5.1f}%")

    print()
    print("=" * 72)
    print("CAST COMPARISON — dividend and tax (normalized, honest)")
    print("=" * 72)
    print(f"{'N':>4} {'bonus roror':>12} {'bonus s64':>12} "
          f"{'tax roror':>10} {'tax s64':>10}")
    roror_bonus = {1: 0.005, 2: 0.010, 4: 0.0255, 8: 0.0657,
                   16: 0.0, 32: -0.403}
    roror_tax = {1: 0.9997, 2: 0.9976, 4: 0.9920, 8: 0.9738,
                 16: 0.9164, 32: 0.7864}
    for n in NS:
        b = rows[n]["veq_f32"] - n
        print(f"{n:4d} {roror_bonus[n]:12.3f} {b:12.3f} "
              f"{roror_tax[n]:10.4f} {rows[n]['tax']:10.4f}")
    print(f"\noverlap duty at N=32: roror ~= 100% (240 s voices at 1.3 s "
          f"spacing), s64 = {duty32*100:.1f}%")

    rows["_meta"] = {"voices": list(S64), "interval": INTERVAL,
                     "normalized": True, "duty32": duty32}
    with open(os.path.join(out, "censuscast2-report.json"), "w") as fj:
        json.dump(rows, fj, indent=1, default=str)
    print(f"\nreport -> {os.path.join(out, 'censuscast2-report.json')}")


if __name__ == "__main__":
    main()
