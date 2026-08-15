#!/usr/bin/env python3
"""Session 70, experiment 1c: REACH THE WALL — N=160 and N=192.

The conditional (full-census-window) clip fraction hit 45.34% at N=128;
the ceiling fit 1-exp(kN) predicts the 50% crossing at N ~= 139.  Build
two more census points past the crossing to CONFIRM it instead of
extrapolating, then merge into the conditional report.

Usage: python3 session70_taxwall3.py <session66_dir> <taxwall_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session68_census import read_f32, write_wav
import analyze_conservation as AC
from session70_taxwall import staircase_metrics

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
NS = (160, 192)
INTERVAL = 1.3


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    d = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/taxwall"
    os.makedirs(d, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())
    ref_energy = AC.analyze(cast[1], 1.0)["total_energy"]

    rows = json.load(open(os.path.join(d, "taxwall-report.json")))
    for N in NS:
        voices = (cast * (N // len(cast) + 1))[:N]
        base = os.path.join(d, f"taxwall-N{N:03d}")
        _, clip_frac, flat_frac, rail_e = staircase_metrics(
            voices, base + "-s16.wav", base + "-f32.wav", INTERVAL, sr)
        e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
        rows[str(N)] = {"veq_s16": e16 / ref_energy, "veq_f32": e32 / ref_energy,
                        "tax": float(e16 / e32), "clip_frac": clip_frac,
                        "flat_frac": flat_frac,
                        "wall_ratio": float(flat_frac / clip_frac)
                        if clip_frac > 0 else 0.0,
                        "rail_share": float(rail_e / e16) if e16 > 0 else 0.0,
                        "rms_db_s16": AC.analyze(base + "-s16.wav", 1.0)["rms_db"]}
        print(f"N={N:3d}: veq(s16) {rows[str(N)]['veq_s16']:7.3f}  "
              f"veq(f32) {rows[str(N)]['veq_f32']:7.3f}  "
              f"tax {rows[str(N)]['tax']:.4f}  "
              f"clip {clip_frac*100:5.2f}%")

    # rebuild the global report with integer keys where possible
    merged = {}
    for k, v in rows.items():
        try:
            merged[int(k)] = v
        except (ValueError, TypeError):
            merged[k] = v
    with open(os.path.join(d, "taxwall-report.json"), "w") as fj:
        json.dump(merged, fj, indent=1, default=str)
    print(f"\nmerged report -> {os.path.join(d, 'taxwall-report.json')}")


if __name__ == "__main__":
    main()
