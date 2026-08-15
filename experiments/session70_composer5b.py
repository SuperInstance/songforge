#!/usr/bin/env python3
"""Session 70, experiment 3b: THE SIGN IS A PRICE.

v5's prescreen showed the seam sign FLIPS with the rented width: every
handoff reads strongly NEGATIVE at X=0.25 (corr -0.85..-0.88 -> DIP) but
POSITIVE at X=1.0 (h0 +0.52, h1 +0.45, h2 +0.04 -> BUMP).  If the sign
is a price of X — not a property of the handoff — then renting X=1.0 on
h1 should PLACE A BUMP, and the same address that measured a 32 dB dip
at X=0.25 should measure a bump at X=1.0.

Test: build three relays, one per handoff targeted at X=1.0 (others at
X=0.25), verify with the ghost-proof meter whether the placed sign at
the target address matches the prescreen sign AT THE RENTED WIDTH.

Prediction: sign(h) = sign(corr(h, X_rented)).  If confirmed, v5's
naming rule is: read the sign at the width you rent.

Usage: python3 session70_composer5b.py <session66_dir> <out_dir>
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


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/composer5b"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    envs = [env_db(v)[0] for v in cast]
    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)

    # prescreen at the widths we will rent
    signs = {}
    for i in range(3):
        row = {X: pair_corr(envs[i], envs[i + 1], X, WS) for X in GRID}
        signs[i] = row

    print("=" * 72)
    print("THE SIGN IS A PRICE — rent X=1.0 on each handoff, one at a")
    print("                    time; others stay at X=0.25")
    print("=" * 72)
    report = {}
    for target in (0, 1, 2):
        xs = [0.25, 0.25, 0.25]
        xs[target] = 1.0
        starts = [0.0]
        for j in range(3):
            starts.append(starts[j] + durs[j] - xs[j])
        a, _ = anchors(durs, xs)
        T = a[target] - xs[target] / 2.0

        f = os.path.join(out, f"rentX1-h{target}.wav")
        build_relay_vx(cast, f, xs)
        pd, _ = env_db(f)
        b = body_level(pd, WS, *body)

        t_dip, v_dip, holes = ghost_proof_min(pd, envs, starts, target, WS, T,
                                              body_db=b)
        t_bump, v_bump, _ = ghost_proof_max(pd, envs, starts, target, WS, T,
                                            body_db=b)
        depth = b - v_dip
        bump = b - v_bump
        corr = signs[target][1.0]
        read = "BUMP" if corr > 0 else "DIP"
        placed = "BUMP" if bump >= depth else "DIP"
        match = read == placed
        print(f"h{target}: X=1.0 corr {corr:+.3f} read {read:4s}  "
              f"placed {placed:4s}  dip {depth:5.1f} dB  "
              f"bump {bump:5.1f} dB  {'MATCH' if match else 'MISMATCH'}")
        report[f"h{target}"] = {"corr": round(float(corr), 3),
                                "read": read, "placed": placed,
                                "match": bool(match),
                                "depth": round(float(depth), 1),
                                "bump": round(float(bump), 1),
                                "dip_err": round(float(t_dip - T), 3),
                                "bump_err": round(float(t_bump - T), 3),
                                "address": round(T, 3),
                                "holes": holes}

    with open(os.path.join(out, "composer5b-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'composer5b-report.json')}")


if __name__ == "__main__":
    main()
