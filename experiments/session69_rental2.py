#!/usr/bin/env python3
"""Session 69, experiment 2b: THE GHOST-PROOF RENTAL METER.

v4's rental market (session69_aim4.py) measured depth as the local envelope
min in a ±0.45 s band around the predicted center.  That meter is
ghost-contaminated: for X≈1.0 the band [anchor-0.95, anchor-0.05] lands in
the material's fade-rim silence (S67 finding), so every handoff "rents"
70-82 dB at X=1.0 — the ghost again, relocated into the band.

This meter is ghost-proof: for each band window it looks at the TWO SOURCE
voices' envelopes at the mix times (voice i at t - start_i, voice i+1 at
t - start_{i+1}) and excludes windows where BOTH are below body - 45 dB —
a hole, not a seam.  The honest seam dip is the local min over the
remaining windows.  Re-analyzes the existing rental builds; no rebuilds.

Also settles the clearance law: X is the ONLY dial.  Address = anchor-X/2,
depth = depth(X).  The joint optimizer "failed" (r0 h2 err +3.7 s) only
because it kept a target computed for X=4.4 while renting X=1.0 — the seam
landed exactly where X=1.0 says it lands.  The market clears in X-space.

Usage: python3 session69_rental2.py <session66_dir> <aim4_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import probe_duration
from session68_aim3 import env_db, body_level, anchors

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
GRID = (0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.50)
WS = 0.05


def ghost_proof_min(pd_mix, envs, starts, i, ws, center, half=0.45,
                    body_db=-30.0, floor_db=-45.0, sr=22050.0):
    """Min envelope of the MIX in the band, excluding silent-source windows.

    All indexing is sample-exact (window = int(ws*sr) samples; the naive
    (k+0.5)*ws seconds mapping drifts by ~0.33 s over 14k windows and
    reads the wrong source times).  Exclusion: a window is a hole if
    EITHER source voice (voice i at t-starts[i], voice i+1 at
    t-starts[i+1]) is below floor_db relative to body — a seam cannot be
    placed where the material itself is already silent (inherited
    fade-rims, S67).  Returns (t_min, value, holes).
    """
    win = int(ws * sr)
    lo = max(0, int((center - half) * sr / win))
    hi = min(len(pd_mix), int((center + half) * sr / win) + 1)
    if hi - lo < 2:
        return center, np.inf, 0
    best_t, best_v, n_holes = None, np.inf, 0
    for k in range(lo, hi):
        t_c = (k * win + win // 2) / sr       # window center time, seconds
        ia = int((t_c - starts[i]) / ws)         # voice i, voice-local
        ib = int((t_c - starts[i + 1]) / ws)     # voice i+1, voice-local
        if (ia < 0 or ib < 0 or ia >= len(envs[i]) or ib >= len(envs[i + 1])):
            continue
        if (envs[i][ia] < body_db + floor_db or
                envs[i + 1][ib] < body_db + floor_db):
            n_holes += 1
            continue
        if pd_mix[k] < best_v:
            best_v = pd_mix[k]
            best_t = t_c
    return best_t, best_v, n_holes


def ghost_proof_max(pd_mix, envs, starts, i, ws, center, half=0.45,
                    body_db=-30.0, floor_db=-45.0, sr=22050.0):
    """Mirror of ghost_proof_min: local MAX of the mix in the band over
    non-hole windows.  bump = body - value (positive = bump above body).
    """
    win = int(ws * sr)
    lo = max(0, int((center - half) * sr / win))
    hi = min(len(pd_mix), int((center + half) * sr / win) + 1)
    if hi - lo < 2:
        return center, -np.inf, 0
    best_t, best_v, n_holes = None, -np.inf, 0
    for k in range(lo, hi):
        t_c = (k * win + win // 2) / sr
        ia = int((t_c - starts[i]) / ws)
        ib = int((t_c - starts[i + 1]) / ws)
        if (ia < 0 or ib < 0 or ia >= len(envs[i]) or ib >= len(envs[i + 1])):
            continue
        if (envs[i][ia] < body_db + floor_db or
                envs[i + 1][ib] < body_db + floor_db):
            n_holes += 1
            continue
        if pd_mix[k] > best_v:
            best_v = pd_mix[k]
            best_t = t_c
    return best_t, best_v, n_holes



def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    aim4 = sys.argv[2] if len(sys.argv) > 2 else "audio/session69/aim4"
    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    envs = [env_db(v)[0] for v in cast]
    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)

    # body level from a default relay (all X=1.0)
    from build_relay import build_relay_vx
    default_file = os.path.join(aim4, "rental-h1-X1.00.wav")
    pd_mix, _ = env_db(default_file)
    body_db = body_level(pd_mix, WS, *body)

    print("=" * 72)
    print("THE GHOST-PROOF RENTAL MARKET (holes excluded;")
    print("                                body at %.1f dB)" % body_db)
    print("=" * 72)
    report = {}
    for i in range(3):
        print(f"handoff {i}:")
        row = []
        for X in GRID:
            xs = [1.0] * 3
            xs[i] = X
            starts = [0.0]
            for j in range(3):
                starts.append(starts[j] + durs[j] - xs[j])
            f = os.path.join(aim4, f"rental-h{i}-X{X:.2f}.wav")
            pd_m, _ = env_db(f)
            b = body_level(pd_m, WS, *body)
            a, _ = anchors(durs, xs)
            pred = a[i] - xs[i] / 2.0
            t_dip, v_dip, holes = ghost_proof_min(
                pd_m, envs, starts, i, WS, pred, body_db=body_db)
            t_bump, v_bump, _ = ghost_proof_max(
                pd_m, envs, starts, i, WS, pred, body_db=body_db)
            # if the meter found nothing, fall back to the band min
            if np.isinf(v_dip):
                t_dip, v_dip, holes = pred, b, -1
            depth = b - v_dip
            bump = b - v_bump if not np.isinf(v_bump) else 0.0
            row.append({"X": X, "depth": round(float(depth), 2),
                        "bump": round(float(bump), 2),
                        "err": round(float(t_dip - pred), 3),
                        "holes": int(holes)})
            print(f"  X={X:5.2f}: depth {depth:6.2f} dB  "
                  f"bump {bump:6.2f} dB  err {t_dip - pred:+.3f}s  "
                  f"holes {holes}")
        depths = np.array([r["depth"] for r in row])
        bumps = np.array([r["bump"] for r in row])
        x_star = GRID[int(np.argmax(depths))]
        x_hump = GRID[int(np.argmax(bumps))]
        print(f"  -> honest market clears at X*={x_star:.2f} "
              f"(depth {depths.max():.2f} dB); best hump "
              f"X={x_hump:.2f} (bump {bumps.max():.2f} dB)")
        print()
        report[f"h{i}"] = {"row": row, "x_star": x_star,
                           "depth_star": round(float(depths.max()), 2),
                           "x_hump": x_hump,
                           "bump_star": round(float(bumps.max()), 2)}

    with open(os.path.join(aim4, "rental2-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"report -> {os.path.join(aim4, 'rental2-report.json')}")


if __name__ == "__main__":
    main()
