#!/usr/bin/env python3
"""Session 70, experiment 1b: THE WALL, CONDITIONED ON THE CENSUS.

The global clip fraction saturates (28.6% at N=128) because it is a
MIXTURE over overlap counts: voices enter at i*1.3 s and run ~240 s, so
the fraction of the file where all N voices overlap SHRINKS as N grows
([(N-1)*1.3, 240] gets shorter) while the entry ramp and solo tails
dilute the average.  The honest wall lives in the full-census window —
the region where every voice is present.

This script re-analyzes the existing taxwall builds (f32 = unclipped
truth, s16 = the container) and measures, per N:
  - t_all: the all-overlap window [max(0,(N-1)*I), min(dur0, ...)]
  - clip fraction IN the window     (the true wall density)
  - flat fraction IN the window
  - tax IN the window               (s16 energy / f32 energy there)
  - window share of total time      (how much of the file is census)

Questions:
  - Does the CONDITIONAL clip fraction cross 50%?  At what N?
  - Is the conditional tax curve smooth or does IT break too?
  - What does the wall look like as N -> 128: a ceiling (saturating
    below 1.0) or a takeover (approaching 1.0)?

Usage: python3 session70_taxwall2.py <taxwall_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session68_census import read_f32

NS = None  # read from merged report keys
INTERVAL = 1.3
SR = 22050.0


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "audio/session70/taxwall"
    rows = json.load(open(os.path.join(d, "taxwall-report.json")))
    dur0 = 241.288  # voice duration (roror-div-xx1p0), s
    NS = tuple(sorted(int(k) for k in rows if isinstance(k, int) or
                      str(k).isdigit()))

    print("=" * 72)
    print("THE WALL, CONDITIONED ON THE FULL-CENSUS WINDOW")
    print("=" * 72)
    out = {}
    for N in NS:
        f32 = read_f32(os.path.join(d, f"taxwall-N{N:03d}-f32.wav"))
        s16r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i",
             os.path.join(d, f"taxwall-N{N:03d}-s16.wav"),
             "-f", "f32le", "-ac", "1", "-"],
            capture_output=True)
        s16 = np.frombuffer(s16r.stdout, dtype=np.float32).astype(np.float64)
        n = min(len(f32), len(s16))
        f32, s16 = f32[:n], s16[:n]
        t0 = max(0.0, (N - 1) * INTERVAL)
        t1 = min(dur0, n / SR)
        i0, i1 = int(t0 * SR), int(t1 * SR)
        seg_f, seg_s = f32[i0:i1], s16[i0:i1]
        clip = float((np.abs(seg_f) >= 1.0).mean())
        # s16 -> f32 round trip scales by 1/32768, so the rails land at
        # 32767/32768 = 0.999969, never exactly 1.0.  Measure the wall
        # against the true rail value.
        RAIL = 32767.0 / 32768.0
        flat = float((np.abs(seg_s) >= RAIL - 1e-9).mean())
        e_f = float((seg_f ** 2).sum())
        e_s = float((seg_s ** 2).sum())
        tax = e_s / e_f if e_f > 0 else 0.0
        share = (t1 - t0) / (n / SR)
        out[N] = {"t_all": [round(t0, 1), round(t1, 1)],
                  "clip_in": clip, "flat_in": flat, "tax_in": tax,
                  "win_share": share}
        print(f"N={N:3d}: window [{t0:6.1f},{t1:6.1f}]s ({share*100:4.1f}% "
              f"of file)  clip {clip*100:5.2f}%  flat {flat*100:5.2f}%  "
              f"tax {tax:.4f}")

    # ---- the conditional 50% crossing
    print()
    print("=" * 72)
    print("THE CONDITIONAL WALL CROSSING (clip -> 50% in the census window)")
    print("=" * 72)
    cross = None
    ns = np.array(sorted(NS))
    clips = np.array([out[n]["clip_in"] for n in ns])
    for (n0, c0), (n1, c1) in zip(zip(ns, clips), list(zip(ns, clips))[1:]):
        if c0 < 0.50 <= c1:
            ln = np.log(n1 / n0)
            frac = (0.50 - c0) / (c1 - c0)
            cross = n0 * np.exp(ln * frac)
            print(f"50% conditional clip between N={n0} ({c0*100:.1f}%) "
                  f"and N={n1} ({c1*100:.1f}%) -> N* ~= {cross:.1f}")
            break
    if cross is None:
        # extrapolate: fit clip(N) = 1 - exp(-k*N) ceiling model
        k_fit = np.polyfit(ns, np.log(1 - clips + 1e-12), 1)[0]
        print(f"no crossing in range; ceiling fit 1-exp(-kN): k={k_fit:.4f}, "
              f"clip(256) ~= {1 - np.exp(k_fit*256):.1%}, "
              f"clip(512) ~= {1 - np.exp(k_fit*512):.1%}")
        out["_ceiling"] = {"k": float(k_fit)}
    else:
        out["_cross50"] = round(cross, 2)

    with open(os.path.join(d, "taxwall-conditional-report.json"), "w") as fj:
        json.dump(out, fj, indent=1)
    print(f"\nreport -> {os.path.join(d, 'taxwall-conditional-report.json')}")


if __name__ == "__main__":
    main()
