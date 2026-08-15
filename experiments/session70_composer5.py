#!/usr/bin/env python3
"""Session 70, experiment 3: COMPOSER v5 — THE SIGN-READING AIMER.

v2 aimed, v3 damped, v4 rented.  v4's clearance law: X is the ONLY
dial — it sets the address (anchor - X/2), the depth, AND the sign of
the feature.  S69 found the mirror trade: at X=0.25 handoff 1 rents a
+24 dB BUMP and handoff 2 a +31 dB bump (h2 is a knife-edge: the same
address measures 31 dB dip AND 31 dB bump); handoff 0 stays
anti-correlated (a valley through every listing).  But v4 always asked
for a DIP — it never read the sign first.

v5 reads the sign BEFORE naming the feature:
  1. PRESCREEN: for each handoff, tail x head envelope correlation
     (pair_corr) at the rental grid.  corr < 0 -> the seam will be a
     DIP (fade material anti-correlates -> destructive seam); corr > 0
     -> BUMP (material agrees -> constructive seam).
  2. NAME: the composer names each seam by its predicted sign, not by
     default.  ("This seam is a bump" / "This seam is a dip".)
  3. RENT: X* = the tightest width (market clears at 0.25) for every
     targeted handoff — the sign is free, the width is rented.
  4. VERIFY: build, then measure BOTH dip and bump at the address with
     the ghost-proof meter; the placed feature's sign should match the
     read sign, and |depth| should be the rented magnitude.

Questions:
  - Does the prescreen sign predict the placed sign for ALL handoffs?
    (h0 anti-correlated -> dip; h1/h2 correlated -> bump)
  - Is the sign stable across X, or does it flip with the width (the
    knife-edge is a sign-flip at the same address)?
  - When the composer NAMES the feature correctly, does it also land
    at the right address (err < 50 ms)?

Usage: python3 session70_composer5.py <session66_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay_vx, probe_duration
from session68_aim3 import env_db, body_level, anchors, pair_corr
from session69_rental2 import ghost_proof_min, ghost_proof_max

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
GRID = (0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.50)
WS = 0.05
XSTAR = 0.25  # the market-clearing width (S69 rental result)


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/composer5"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    envs = [env_db(v)[0] for v in cast]
    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)

    # ---- 1. PRESCREEN: the sign of every seam at every width
    print("=" * 72)
    print("THE SIGN PRESCREEN (tail x head envelope correlation)")
    print("=" * 72)
    signs = {}
    for i in range(3):
        tail_env = envs[i]
        head_env = envs[i + 1]
        row = {}
        for X in GRID:
            c = pair_corr(tail_env, head_env, X, WS)
            row[X] = round(c, 4)
        signs[i] = row
        named = "DIP" if row[XSTAR] < 0 else "BUMP"
        print(f"h{i}: X=0.25 corr {row[XSTAR]:+.3f} -> seam named {named}")
        print(f"     grid: " + "  ".join(
            f"{X}:{row[X]:+.2f}" for X in GRID))

    # ---- 2. RENT: X* = tightest width everywhere; derive addresses
    xs = [XSTAR] * 3
    starts = [0.0]
    for j in range(3):
        starts.append(starts[j] + durs[j] - xs[j])
    a, _ = anchors(durs, xs)
    addresses = {i: a[i] - xs[i] / 2.0 for i in range(3)}

    # ---- 3. BUILD the named relay
    f = os.path.join(out, "sign-reader-relay.wav")
    build_relay_vx(cast, f, xs)
    pd, _ = env_db(f)
    b = body_level(pd, WS, *body)
    print(f"\nbody level: {b:.1f} dB  xs = {xs}")

    # ---- 4. VERIFY: dip AND bump at each address; sign must match
    print()
    print("=" * 72)
    print("THE VERIFICATION (placed feature vs read sign)")
    print("=" * 72)
    report = {}
    for i in range(3):
        T = addresses[i]
        t_dip, v_dip, holes = ghost_proof_min(pd, envs, starts, i, WS, T,
                                              body_db=b)
        t_bump, v_bump, _ = ghost_proof_max(pd, envs, starts, i, WS, T,
                                            body_db=b)
        depth = b - v_dip
        bump = b - v_bump
        read_sign = "DIP" if signs[i][XSTAR] < 0 else "BUMP"
        # placed feature: the stronger of the two (bigger |dB from body|)
        placed = "DIP" if depth >= bump else "BUMP"
        match = read_sign == placed
        print(f"h{i}: read {read_sign:4s}  placed {placed:4s}  "
              f"dip {depth:5.1f} dB @ {t_dip:7.3f}s (err {t_dip-T:+.3f})  "
              f"bump {bump:5.1f} dB @ {t_bump:7.3f}s (err {t_bump-T:+.3f})  "
              f"{'MATCH' if match else 'MISMATCH'}  holes {holes}")
        report[f"h{i}"] = {"read_sign": read_sign, "placed": placed,
                           "match": bool(match),
                           "depth": round(float(depth), 1),
                           "bump": round(float(bump), 1),
                           "dip_at": round(float(t_dip), 3),
                           "bump_at": round(float(t_bump), 3),
                           "dip_err": round(float(t_dip - T), 3),
                           "bump_err": round(float(t_bump - T), 3),
                           "address": round(T, 3),
                           "corr_grid": signs[i]}

    with open(os.path.join(out, "composer5-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'composer5-report.json')}")


if __name__ == "__main__":
    main()
