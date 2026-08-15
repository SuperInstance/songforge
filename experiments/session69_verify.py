#!/usr/bin/env python3
"""Session 69, experiment 2c: THE CLEARANCE VERIFICATION.

The rental market (session69_rental2.py) clears at X=0.25 for every
handoff: tightest crossfade = deepest seam (26-32 dB) AND tightest
position (err < 50 ms).  This script closes the loop the way the v4
optimizer should: choose X* from the market, DERIVE the target address
from X* (T = anchor - X*/2 — the address is a price of X, not an
independent wish), build, and verify position + depth with the
ghost-proof meter.

Also verifies the hump at the same address (the knife-edge).

Usage: python3 session69_verify.py <session66_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay_vx, probe_duration
from session68_aim3 import env_db, body_level, anchors
from session69_rental2 import ghost_proof_min, ghost_proof_max

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
WS = 0.05


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session69/verify"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    envs = [env_db(v)[0] for v in cast]
    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)

    # market clears at X*=0.25; rent it on the two targeted handoffs,
    # leave the middle handoff at the default 1.0
    XSTAR = 0.25
    xs = [XSTAR, 1.0, XSTAR]
    starts = [0.0]
    for j in range(3):
        starts.append(starts[j] + durs[j] - xs[j])
    a, _ = anchors(durs, xs)

    f = os.path.join(out, "joint-r-final.wav")
    build_relay_vx(cast, f, xs)
    pd, _ = env_db(f)
    b = body_level(pd, WS, *body)

    report = {}
    print("=" * 72)
    print("THE CLEARANCE — X* = 0.25 on handoffs 0 and 2;")
    print("              address derived from X (T = anchor - X*/2)")
    print("=" * 72)
    for i in (0, 2):
        T = a[i] - xs[i] / 2.0            # the address X rents
        pred = T
        t_dip, v_dip, holes = ghost_proof_min(pd, envs, starts, i, WS, pred,
                                              body_db=b)
        t_bump, v_bump, _ = ghost_proof_max(pd, envs, starts, i, WS, pred,
                                            body_db=b)
        depth = b - v_dip
        bump = b - v_bump
        print(f"h{i}: X={xs[i]:.2f} address={T:8.3f}s  "
              f"dip@{t_dip:8.3f}s (err {t_dip - T:+.3f}s)  "
              f"depth {depth:5.1f} dB  |  bump@{t_bump:8.3f}s "
              f"({bump:+.1f} dB)  holes {holes}")
        report[f"h{i}"] = {"x": xs[i], "address": round(T, 3),
                           "dip_at": round(float(t_dip), 3),
                           "err": round(float(t_dip - T), 3),
                           "depth": round(float(depth), 1),
                           "bump": round(float(bump), 1)}

    with open(os.path.join(out, "verify-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'verify-report.json')}")


if __name__ == "__main__":
    main()
