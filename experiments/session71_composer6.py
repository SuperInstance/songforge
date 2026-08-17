#!/usr/bin/env python3
"""Session 71, experiment 3: COMPOSER v6 — RENT THE SIGN.

v5 found the sign is a PRICE: at the tight width (X=0.25) the prescreen
reads the seam honestly (corr < 0 -> DIP placed, 3/3); at wide X it
reads the BODY (corr > 0 -> BUMP predicted, DIP placed, 0/3 — ghost
signs).  The corr_grid from v5 shows the prescreen CORR CROSSES ZERO
between X=0.5 (h0: -0.309) and X=1.0 (h0: +0.523) — there is a price
where the sign is undefined, a knife-edge in X-space.

v6 asks: CAN YOU BUY THE SIGN?  If the prescreen is a price list, then
  - rent X just BELOW the crossing  -> buy a DIP
  - rent X just ABOVE the crossing  -> buy a BUMP
and the placed feature should flip with the rented width.  If instead
the crossing is where the prescreen begins to lie (the body starts
dominating the window), then renting above the crossing buys nothing —
the placed sign stays DIP and the crossing is the last honest price.

Design (per handoff):
  1. PRICE LIST: corr(X) on a fine grid 0.05..1.00, step 0.05.
  2. CROSSING: X* where corr changes sign (linear interpolation).
  3. RENT: X_lo = X* - 0.10 (negative side -> DIP), X_hi = X* + 0.10
     (positive side -> BUMP).
  4. BUILD two relays (all handoffs on the same side), verify with the
     ghost-proof meter: dip AND bump at every address.
  5. VERDICT: does the sign flip with the side?  Per-handoff and as a
     relay (all-3 agreement).

Usage: python3 session71_composer6.py <session66_dir> <out_dir>
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
GRID = tuple(round(0.05 * k, 2) for k in range(1, 21))  # 0.05..1.00
WS = 0.05
SIDE = 0.10


def find_crossing(rows):
    """Linear interpolation of the corr sign change; None if none.

    Skips the +0.000 sentinel: pair_corr returns exactly 0.0 when the
    window has n<5 samples (X < 0.25) — that's NO READING, not a sign.
    Only crossings between two non-sentinel readings count.
    """
    vals = [(x, c) for x, c in rows if c != 0.0]
    for k in range(len(vals) - 1):
        x0, c0 = vals[k]
        x1, c1 = vals[k + 1]
        if c0 * c1 < 0:
            return float(x0 + (0 - c0) * (x1 - x0) / (c1 - c0))
    return None


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session71/composer6"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    envs = [env_db(v)[0] for v in cast]
    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)

    print("=" * 72)
    print("THE PRICE LIST (prescreen corr vs X, fine grid)")
    print("=" * 72)
    crossings = {}
    price_lists = {}
    for i in range(3):
        row = []
        for X in GRID:
            c = pair_corr(envs[i], envs[i + 1], X, WS)
            row.append((X, c))
        price_lists[i] = row
        xstar = find_crossing(row)
        crossings[i] = xstar
        sign_lo = "DIP" if [c for _, c in row if _ <= (xstar or 1.0)][-1] < 0 else "BUMP"
        print(f"h{i}: crossing X* = {xstar if xstar else 'NONE':>5}  "
              f"X=0.25 {row[4][1]:+.3f}  X=0.5 {row[9][1]:+.3f}  "
              f"X=1.0 {row[19][1]:+.3f}")

    # ---- RENT both sides and build
    results = {}
    for side, tag in (("lo", "dip"), ("hi", "bump")):
        xs = []
        for i in range(3):
            xstar = crossings[i]
            if xstar is None:
                xs.append(0.25 if side == "lo" else 1.00)
            else:
                xs.append(max(0.05, min(1.00, xstar + (SIDE if side == "hi"
                                                       else -SIDE))))
        starts = [0.0]
        for j in range(3):
            starts.append(starts[j] + durs[j] - xs[j])
        a, _ = anchors(durs, xs)
        addresses = {i: a[i] - xs[i] / 2.0 for i in range(3)}

        f = os.path.join(out, f"rent-{tag}.wav")
        build_relay_vx(cast, f, xs)
        pd, _ = env_db(f)
        b = body_level(pd, WS, *body)
        print(f"\n[{tag.upper()} side] xs = {[round(x,2) for x in xs]}  "
              f"body {b:.1f} dB")

        results[side] = {"xs": [round(x, 2) for x in xs], "body": round(b, 1)}
        for i in range(3):
            T = addresses[i]
            t_dip, v_dip, holes = ghost_proof_min(pd, envs, starts, i, WS, T,
                                                  body_db=b)
            t_bump, v_bump, _ = ghost_proof_max(pd, envs, starts, i, WS, T,
                                                body_db=b)
            if t_dip is None:
                t_dip, v_dip = T, b  # all-hole band: no seam readable
            if t_bump is None:
                t_bump, v_bump = T, b
            depth = b - v_dip
            bump = b - v_bump
            placed = "DIP" if depth >= bump else "BUMP"
            wanted = "DIP" if side == "lo" else "BUMP"
            results[side][f"h{i}"] = {
                "x": round(xs[i], 2), "address": round(T, 3),
                "depth": round(float(depth), 1), "bump": round(float(bump), 1),
                "placed": placed, "wanted": wanted,
                "match": placed == wanted,
                "dip_err": round(float(t_dip - T), 3),
                "bump_err": round(float(t_bump - T), 3)}
            print(f"  h{i}: wanted {wanted:4s} placed {placed:4s}  "
                  f"dip {depth:5.1f}  bump {bump:5.1f}  "
                  f"{'MATCH' if placed == wanted else 'no'}  "
                  f"holes {holes}")

    # ---- VERDICT
    lo_m = sum(1 for i in range(3) if results["lo"][f"h{i}"]["match"])
    hi_m = sum(1 for i in range(3) if results["hi"][f"h{i}"]["match"])
    print()
    print("=" * 72)
    print("THE VERDICT: can you buy the sign?")
    print("=" * 72)
    print(f"  rent below crossing (DIP):  {lo_m}/3 matches")
    print(f"  rent above crossing (BUMP): {hi_m}/3 matches")
    if lo_m == 3 and hi_m == 3:
        v = "THE SIGN IS FOR SALE: the prescreen crossing is a true price;"
        v += " rent below for a dip, above for a bump, and the feature flips."
    elif lo_m == 3 and hi_m == 0:
        v = "THE CROSSING IS THE LAST HONEST PRICE: below it the sign is"
        v += " readable (dip), above it the prescreen reads the body and"
        v += " sells bumps that never arrive — the ghost sign has a price."
    else:
        v = f"PARTIAL: dip-side {lo_m}/3, bump-side {hi_m}/3 — the price"
        v += " list is not uniform across handoffs."
    print(f"  {v}")

    report = {"crossings": {str(i): crossings[i] for i in range(3)},
              "price_lists": {str(i): [[x, c] for x, c in price_lists[i]]
                              for i in range(3)},
              "results": results, "verdict": v}
    with open(os.path.join(out, "composer6-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'composer6-report.json')}")


if __name__ == "__main__":
    main()
