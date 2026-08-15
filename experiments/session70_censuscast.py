#!/usr/bin/env python3
"""Session 70, experiment 2: THE CENSUS WITH A DIFFERENT CAST.

S69's census (and the tax curve) used the layer-3 roror cast — four
voices that are FOUR NEAR-IDENTICAL RENDITIONS of the same phrase
("roror...") at different crossfade settings.  The dividend sign-flip at
N=32 (bonus = -0.403, the crowd undercounts itself) was measured on that
near-identical cast.  Is the sign-flip a property of the CENSUS (any
crowd of overlapping voices) or of the CAST (identical material
agreeing with itself at a lag)?

Test: run the same staircase census (interval 1.3 s, N = 1..32, s16 and
f32 containers) on the session-64 cast — FOUR GENUINELY DIFFERENT
SINGERS (lessac, norman, joe, amy; distinct speakers, distinct
utterances).  Compare:
  - veq(N), bonus(N) = veq_f32 - N: does the dividend stay positive /
    ~0 (heterogeneous crowd counts honestly) or flip negative here too?
  - container tax curve vs the roror cast: is the s16 tax a cast
    property or a census property?
  - the twin hypothesis: the roror cast is 4 copies of one phrase; the
    s64 cast is 4 independent voices — the SAME voices repeat at N>4
    (cast cycled), but their lags are now independent material.

Usage: python3 session70_censuscast.py <session64_dir> <out_dir>
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


def main():
    s64 = sys.argv[1] if len(sys.argv) > 1 else "audio/session64"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/censuscast"
    os.makedirs(out, exist_ok=True)

    cast = [os.path.join(s64, n) for n in S64]
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())

    ref_energy = AC.analyze(cast[0], 1.0)["total_energy"]

    print("=" * 72)
    print("THE CENSUS WITH A DIFFERENT CAST (s64: 4 distinct singers,")
    print("      interval 1.3 s, N = 1..32)")
    print("=" * 72)
    rows = {}
    for N in NS:
        voices = (cast * (N // len(cast) + 1))[:N]
        base = os.path.join(out, f"cast-N{N:02d}")
        from session70_taxwall import staircase_metrics
        _, clip_frac, flat_frac, _ = staircase_metrics(
            voices, base + "-s16.wav", base + "-f32.wav", INTERVAL, sr)
        e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
        tax = e16 / e32
        veq16 = e16 / ref_energy
        veq32 = e32 / ref_energy
        rows[N] = {"veq_s16": veq16, "veq_f32": veq32,
                   "tax": float(tax), "clip_frac": clip_frac,
                   "flat_frac": flat_frac}
        print(f"N={N:2d}: veq(s16) {veq16:7.3f}  veq(f32) {veq32:7.3f}  "
              f"tax {tax:.4f}  clip {clip_frac*100:5.2f}%")

    # ---- the dividend on the heterogeneous cast
    print()
    print("=" * 72)
    print("THE CENSUS DIVIDEND (bonus = veq_f32 - N) — heterogeneous cast")
    print("=" * 72)
    ns = np.array(sorted(rows))
    bonus = {n: rows[n]["veq_f32"] - n for n in ns}
    for n in ns:
        print(f"N={n:2d}: bonus {bonus[n]:+.4f}")
    # fit exponent on positive points (like S69)
    fit_n = np.array([n for n in ns if bonus[n] > 0])
    fit_b = np.array([bonus[n] for n in fit_n])
    if len(fit_n) >= 2:
        logb, logn = np.log(fit_b), np.log(fit_n)
        b_fit, a_fit = np.polyfit(logn, logb, 1)
        print(f"fit (positive side): bonus = {np.exp(a_fit):.4f} * N^"
              f"{b_fit:.3f}   (S69 roror cast: a=0.0045, b=1.273)")
        rows["_fit"] = {"a": float(np.exp(a_fit)), "b": float(b_fit)}

    # ---- compare with the S69 roror cast (hard-coded series)
    print()
    print("=" * 72)
    print("CAST COMPARISON (roror near-identical vs s64 distinct singers)")
    print("=" * 72)
    roror_bonus = {1: 0.005, 2: 0.010, 4: 0.0255, 8: 0.0657,
                   16: 0.0, 32: -0.403}
    print(f"{'N':>4} {'bonus roror':>12} {'bonus s64':>12} "
          f"{'tax roror':>10} {'tax s64':>10}")
    for n in ns:
        rb = roror_bonus.get(int(n), float("nan"))
        print(f"{n:4d} {rb:12.3f} {bonus[n]:12.3f} "
              f"{0.0:10.3f} {rows[n]['tax']:10.4f}")

    with open(os.path.join(out, "censuscast-report.json"), "w") as fj:
        json.dump(rows, fj, indent=1, default=str)
    print(f"\nreport -> {os.path.join(out, 'censuscast-report.json')}")


if __name__ == "__main__":
    main()
