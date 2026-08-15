#!/usr/bin/env python3
"""Session 69, experiment 2: THE RENTAL MARKET — composer-loop v4.

S68 v3 established: POSITION IS FREE, DEPTH IS RENTED.  A handoff's
silence can be aimed to any address, but how DEEP it lands is set by the
material's correlation at that width (X) — the material rents out only
what its envelope will allow.  v4 makes this the optimizer:

  A. THE RENTAL CURVE — instead of trusting the correlation PROXY, build
     the relay for a grid of X per handoff and MEASURE the placed depth at
     the predicted center (anchor - X/2, ±0.45 s band).  depth(X) is the
     material's rental price list.  The joint optimizer picks
     X* = argmax depth(X) for each targeted handoff.
  B. THE HUMP AIMER — the mirror trade: aim a handoff where tail*head
     correlate POSITIVELY and the seam makes a BUMP instead of a dip.
     Can a hump be placed as reliably as a dip?  depth goes negative;
     report bump height (body - local_max) and position error.

Questions:
  - Is depth(X) a smooth market or a jagged one with teeth (S65's
    resonance teeth at X = 0.25/3.25/5.25)?
  - Does the argmax-of-rental choice converge in ONE round (position and
    depth both correct), unlike v2's ringing and v3's shallow handoff 2?
  - Is the hump as aimable as the dip (same |depth|, same error)?

Usage: python3 session69_aim4.py <session66_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay_vx, probe_duration
import analyze_conservation as AC
from session68_aim3 import env_db, local_min, body_level, anchors, solve_xs

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
XMIN, XMAX = 0.05, 5.50
GRID = (0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.50)


def local_max(pd, ws, center, half=0.45):
    """Max envelope in [center-half, center+half]; mirror of local_min."""
    lo = max(0, int((center - half) / ws))
    hi = min(len(pd), int((center + half) / ws) + 1)
    if hi - lo < 2:
        return center, np.inf
    seg = pd[lo:hi]
    i = int(np.argmax(seg))
    return (lo + i + 0.5) * ws, float(seg[i])


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session69/aim4"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    ws = 0.05
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())

    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)

    report = {"rental_curves": {}, "hump_curves": {}, "optimizer": {}}

    # ---------- A. THE RENTAL CURVES (measured, not proxied) ----------
    print("=" * 72)
    print("THE RENTAL MARKET — depth(X) measured at the predicted center,")
    print("                  for every handoff, grid X in 0.25 .. 5.50")
    print("=" * 72)
    rental = {}
    hump = {}
    for i in range(3):          # handoff i (3 handoffs in a 4-voice relay)
        row_r, row_h = [], []
        for X in GRID:
            xs = [1.0] * 3
            xs[i] = X
            f = os.path.join(out, f"rental-h{i}-X{X:.2f}.wav")
            build_relay_vx(cast, f, xs)
            pd, _ = env_db(f)
            b = body_level(pd, ws, *body)
            a, _ = anchors(durs, xs)
            pred = a[i] - xs[i] / 2.0
            t_dip, v_dip = local_min(pd, ws, pred)
            t_bump, v_bump = local_max(pd, ws, pred)
            d_dip = b - v_dip
            d_bump = b - v_bump        # positive = bump ABOVE body
            row_r.append({"X": X, "depth": round(d_dip, 2),
                          "err": round(t_dip - pred, 3),
                          "center": round(pred, 3)})
            row_h.append({"X": X, "bump": round(d_bump, 2),
                          "err": round(t_bump - pred, 3)})
            print(f"h{i} X={X:5.2f}: dip {d_dip:6.2f} dB "
                  f"(err {t_dip - pred:+.3f}s) | "
                  f"bump {d_bump:6.2f} dB (err {t_bump - pred:+.3f}s)")
        rental[i] = row_r
        hump[i] = row_h
        depths = np.array([r["depth"] for r in row_r])
        bumps = np.array([r["bump"] for r in row_h])
        x_star = GRID[int(np.argmax(depths))]
        print(f"  -> h{i} best RENTAL: X*={x_star:.2f} "
              f"(depth {depths.max():.2f} dB); best HUMP: "
              f"X={GRID[int(np.argmax(bumps))]:.2f} "
              f"(bump {bumps.max():.2f} dB)")
        print()
    report["rental_curves"] = rental
    report["hump_curves"] = hump

    # ---------- B. THE JOINT OPTIMIZER (one round, verify) ----------
    print("=" * 72)
    print("THE JOINT OPTIMIZER — target T0/T2, X* from the rental market,")
    print("                        verify position AND depth, 2 rounds")
    print("=" * 72)
    a0, _ = anchors(durs, [1.0] * 3)
    T0 = round(a0[0] - 1.60, 2)
    xs = solve_xs(durs, {0: T0})
    a_now, _ = anchors(durs, xs)
    T2 = round(a_now[2] - 2.20, 2)
    # X*: pick the grid X whose rental depth is largest for that handoff
    xstar = {}
    for i, T in ((0, T0), (2, T2)):
        depths = np.array([r["depth"] for r in rental[i]])
        xstar[i] = GRID[int(np.argmax(depths))]
    print(f"targets: h0 -> T={T0}s (X*={xstar[0]:.2f})   "
          f"h2 -> T={T2}s (X*={xstar[2]:.2f})")

    xs_cur = [1.0] * 3
    for i, X in xstar.items():
        xs_cur[i] = X
    for rnd in range(2):
        f = os.path.join(out, f"joint-r{rnd}.wav")
        build_relay_vx(cast, f, xs_cur)
        pd, _ = env_db(f)
        b = body_level(pd, ws, *body)
        line = []
        for i, T in ((0, T0), (2, T2)):
            a, _ = anchors(durs, xs_cur)
            pred = a[i] - xs_cur[i] / 2.0
            t_dip, v_dip = local_min(pd, ws, pred)
            depth = b - v_dip
            err = t_dip - T
            line.append(f"r{rnd} h{i}: X={xs_cur[i]:.2f} placed@{t_dip:7.3f}s "
                        f"err {err:+.3f} depth {depth:5.1f} dB "
                        f"[target X* {xstar[i]:.2f}]")
            print(line[-1])
            report["optimizer"].setdefault(f"r{rnd}", {})[f"h{i}"] = {
                "x": round(xs_cur[i], 3), "err": round(err, 3),
                "depth": round(depth, 1)}
        # damped correction on POSITION only; depth is rented as-is
        for i, T in ((0, T0), (2, T2)):
            a, _ = anchors(durs, xs_cur)
            pred = a[i] - xs_cur[i] / 2.0
            t_dip, _ = local_min(pd, ws, pred)
            xs_cur[i] = min(XMAX, max(XMIN,
                                      xs_cur[i] + 0.5 * (t_dip - T)))

    with open(os.path.join(out, "aim4-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'aim4-report.json')}")


if __name__ == "__main__":
    main()
