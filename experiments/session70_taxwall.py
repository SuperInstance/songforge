#!/usr/bin/env python3
"""Session 70, experiment 1: THE FLAT-TOPPED WALL.

S69 pushed the container-tax curve to N=32: tax 0.9997 -> 0.7864,
clip fraction 0 -> 10.32%, flat-top fraction 0 -> 10.32% (one sample in
ten a wall).  Two open questions:

  A. Where is the WALL?  At what N does the clip fraction cross 50% —
     the census majority pinned flat against the rails?  From the S69
     series (0.74% @ N=8, 3.32% @ 16, 10.32% @ 32 — ~4.5x per doubling)
     the crossing should sit around N=64-90.  N=48/64/96/128 bracket it.
  B. SMOOTH OR PHASE TRANSITION?  Is the tax curve a smooth power-law
     descent (tax ~ a*N^-b fits every point) or does it BREAK at the
     wall — a kink where the s16 container's retained share stops
     tracking the f32 census and the wall takes over the ledger?
     Measure both exponents (pre-wall and post-wall fits) and the
     transition band.

Measures per N in {1, 2, 4, 8, 16, 32, 48, 64, 96, 128}:
  - veq(s16), veq(f32)        the census in each container
  - container tax             s16 energy / f32 energy
  - clip fraction             fraction of pre-clip samples |mix| >= 1.0
  - flat-top fraction         fraction of CLIPPED samples pinned at ±1.0
  - wall ratio                flat_frac / clip_frac (1.0 = full wall)
  - rail energy share         energy carried by pinned samples / s16 energy

Also fits the 50% crossing by linear interpolation in log-N.

Usage: python3 session70_taxwall.py <session66_dir> <out_dir>
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

NS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128)
INTERVAL = 1.3


def staircase_metrics(voices, out_s16, out_f32, interval, sr):
    """Same census in s16 and f32; returns dict of wall metrics.

    Accumulates incrementally (voices are 240 s each; N=128 would be
    ~5.4 GB if all loaded at once).
    """
    datas = [read_f32(v) for v in voices]
    lens = [len(d) for d in datas]
    end = max(int(len(d) / sr + i * interval) for i, d in enumerate(datas))
    n = (end + 1) * sr
    mix = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(datas):
        s = int(i * interval * sr)
        mix[s:s + len(d)] += d
    clip_frac = float((np.abs(mix) >= 1.0).mean())
    # rail energy: energy that pinned samples carry after the clip
    clipped = np.abs(mix) >= 1.0
    rail_e = float((mix[clipped] ** 2).sum()) if clipped.any() else 0.0
    write_wav(mix, sr, out_s16, "s16le")
    write_wav(mix, sr, out_f32, "f32le")
    s16 = np.clip(mix, -1.0, 1.0)
    flat_frac = float((np.abs(s16) >= 1.0 - 1e-9).mean())
    return mix, clip_frac, flat_frac, rail_e


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/taxwall"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())

    ref_energy = AC.analyze(cast[1], 1.0)["total_energy"]

    print("=" * 72)
    print("THE FLAT-TOPPED WALL (container-tax curve, N = 1..128)")
    print("=" * 72)
    rows = {}
    for N in NS:
        voices = (cast * (N // len(cast) + 1))[:N]
        base = os.path.join(out, f"taxwall-N{N:03d}")
        _, clip_frac, flat_frac, rail_e = staircase_metrics(
            voices, base + "-s16.wav", base + "-f32.wav", INTERVAL, sr)
        e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
        tax = e16 / e32
        veq16 = e16 / ref_energy
        veq32 = e32 / ref_energy
        rows[N] = {"veq_s16": veq16, "veq_f32": veq32,
                   "tax": float(tax), "clip_frac": clip_frac,
                   "flat_frac": flat_frac,
                   "wall_ratio": float(flat_frac / clip_frac)
                   if clip_frac > 0 else 0.0,
                   "rail_share": float(rail_e / e16) if e16 > 0 else 0.0,
                   "rms_db_s16": AC.analyze(base + "-s16.wav", 1.0)["rms_db"]}
        print(f"N={N:3d}: veq(s16) {veq16:7.3f}  veq(f32) {veq32:7.3f}  "
              f"tax {tax:.4f}  clip {clip_frac*100:5.2f}%  "
              f"flat {flat_frac*100:5.2f}%  rail {rows[N]['rail_share']*100:5.2f}%")

    # ---- the 50% crossing (interpolate in log-N between bracket points)
    print()
    print("=" * 72)
    print("THE WALL CROSSING (clip fraction -> 50%)")
    print("=" * 72)
    ns = np.array(sorted(rows))
    clips = np.array([rows[n]["clip_frac"] for n in ns])
    cross = None
    for (n0, c0), (n1, c1) in zip(zip(ns, clips), list(zip(ns, clips))[1:]):
        if c0 < 0.50 <= c1:
            # linear in log-N
            ln = np.log(n1 / n0)
            frac = (0.50 - c0) / (c1 - c0)
            cross = n0 * np.exp(ln * frac)
            print(f"50% clip crossing between N={n0} ({c0*100:.1f}%) and "
                  f"N={n1} ({c1*100:.1f}%) -> N* ~= {cross:.1f}")
            break
    if cross is None:
        print("no crossing in range (clip fraction never reached 50%)")

    # ---- smooth power law or phase transition?  pre-wall vs post-wall fits
    print()
    print("=" * 72)
    print("SMOOTH OR BREAK?  tax ~ a*N^-b, fitted on both sides of the wall")
    print("=" * 72)
    def fit_power(ns_sel, vals):
        if len(ns_sel) < 2:
            return None
        ln, lv = np.log(np.array(ns_sel)), np.log(np.array(vals))
        b, loga = np.polyfit(ln, lv, 1)
        return float(np.exp(loga)), -float(b)

    pre = [n for n in ns if n <= 32]
    post = [n for n in ns if n >= 48]
    fp = fit_power(pre, [rows[n]["tax"] for n in pre])
    fq = fit_power(post, [rows[n]["tax"] for n in post])
    print(f"pre-wall  (N<=32):  tax = {fp[0]:.4f} * N^-{fp[1]:.3f}" if fp
          else "pre-wall: <2 points")
    print(f"post-wall (N>=48):  tax = {fq[0]:.4f} * N^-{fq[1]:.3f}" if fq
          else "post-wall: <2 points")
    if fp and fq:
        ratio = fp[1] / fq[1]
        print(f"exponent ratio pre/post = {ratio:.2f}  "
              f"({'SMOOTH' if 0.8 < ratio < 1.25 else 'PHASE TRANSITION'} "
              f"signature)")
        rows["_fit"] = {"pre": {"a": fp[0], "b": fp[1]},
                        "post": {"a": fq[0], "b": fq[1]},
                        "ratio": ratio,
                        "cross50": round(cross, 2) if cross else None}

    with open(os.path.join(out, "taxwall-report.json"), "w") as fj:
        json.dump(rows, fj, indent=1, default=str)
    print(f"\nreport -> {os.path.join(out, 'taxwall-report.json')}")


if __name__ == "__main__":
    main()
