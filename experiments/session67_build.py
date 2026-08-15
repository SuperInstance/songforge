#!/usr/bin/env python3
"""Session 67: depth 4, the frozen clock, and the analyzer as composer.

Three experiments, all local (MMX dark until Aug 16):

1. DEPTH 4 — relay of relays of relays of relays (256 voices, 4 deep).
   Cast = the four layer-3 outputs.  Predictions from the S66 series:
   tax X=1.0: 0.735 -> 0.948 -> 0.998 -> ~0.9995 (deficit x~0.25/layer)
   tax X=2.0: 0.547 -> 0.891 -> 0.987 -> ~0.997
   crowd:     2.02 -> 3.14 -> 3.40 -> ~3.5 (ceiling; increment collapsing)
   fate:      0.50 -> 0.56 -> 0.53 -> ~0.5 dB (fixed point)

2. THE FROZEN CLOCK — a composed file's duration is sum(durs) - 3X, so
   the layer-3 cast's duration differences (2.1 / 3.0 / 5.1 s) are the
   SAME as layer-2's.  The internal clock is made of crossfades and
   freezes at first composition.  Test: depth-4 X sweep, targeted at the
   depth-2 teeth (energy dips 2.5-5.0, std teeth 3.0 / 5.25) plus
   controls.  If the clock is frozen, depth 4 rings at the same X.

3. THE ANALYZER BECOMES THE COMPOSER — per handoff, compute the
   envelope-correlation of voice i's tail vs voice i+1's head as a
   function of X (0..5.5, 0.05 steps).  Choose X_i at each pair's
   correlation MINIMUM (predicted energy dip) -> relay-chosen-dips;
   at each pair's MAXIMUM (predicted hump) -> relay-chosen-humps.
   Analysis drives synthesis: the dips are placed, not found.

Usage: python3 session67_build.py <session66_dir> <out_dir>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay, build_staircase, build_relay_vx
import analyze_conservation as AC


def env_db(path, winsize=0.05):
    """Fine energy envelope (dB), 50 ms windows."""
    data, sr = AC.read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)


def pair_corr(tail_env, head_env, X, ws=0.05):
    """Normalized correlation of the last-X / first-X envelope windows."""
    n = int(X / ws)
    if n < 5:
        return 0.0
    ta = tail_env[-n:] - tail_env[-n:].mean()
    hb = head_env[:n] - head_env[:n].mean()
    denom = np.sqrt((ta ** 2).sum() * (hb ** 2).sum())
    return float((ta * hb).sum() / denom) if denom > 0 else 0.0


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session67"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [
        os.path.join(l3, "roror-div-xx0p3.wav"),
        os.path.join(l3, "roror-div-xx1p0.wav"),
        os.path.join(l3, "roror-div-xx2p0.wav"),
        os.path.join(l3, "roror-conv-x1p0.wav"),
    ]

    # ---- 1. DEPTH 4 ----
    l4 = os.path.join(out, "layer4")
    os.makedirs(l4, exist_ok=True)
    for x in (0.3, 1.0, 2.0):
        label = f"x{str(x).replace('.', 'p')}"
        build_relay(cast, os.path.join(l4, f"rororor-div-{label}.wav"), x)
    cast_conv = [cast[3], cast[2], cast[1], cast[0]]
    build_relay(cast_conv, os.path.join(l4, "rororor-conv-x1p0.wav"), 1.0)
    build_staircase(cast, os.path.join(l4, "staircase-of-everything.wav"), 1.3)
    print(f"depth 4 done -> {l4}")

    # ---- 2. FROZEN-CLOCK TARGETED SWEEP ----
    # depth-2 measured: energy dips at 2.5/3.0/4.5/5.0, std teeth 3.0/5.25
    fc = os.path.join(out, "frozen")
    os.makedirs(fc, exist_ok=True)
    for x in (2.0, 2.75, 3.0, 3.25, 4.0, 5.0, 5.25):
        label = f"x{str(x).replace('.', 'p')}"
        build_relay(cast, os.path.join(fc, f"d4-{label}.wav"), x)
    print(f"frozen-clock sweep: 7 files -> {fc}")

    # ---- 3. ANALYZER AS COMPOSER ----
    print("computing per-handoff envelope correlations (layer-3 cast)...")
    envs = [env_db(v) for v in cast]
    Xs = [round(0.05 * i, 2) for i in range(1, 111)]  # 0.05 .. 5.50
    curves = []  # per handoff: list of (X, corr)
    for a, b in zip(envs, envs[1:]):
        curves.append([(X, pair_corr(a, b, X)) for X in Xs])
    chosen_dips, chosen_humps, chosen_mid = [], [], []
    for ci, curve in enumerate(curves):
        xs = [c[0] for c in curve]
        cs = [c[1] for c in curve]
        i_min = int(np.argmin(cs))
        i_max = int(np.argmax(cs))
        i_mid = int(np.argmin([abs(c - 0.0) for c in cs]))
        xd, xh, xm = xs[i_min], xs[i_max], xs[i_mid]
        chosen_dips.append(xd)
        chosen_humps.append(xh)
        chosen_mid.append(xm)
        print(f"  handoff {ci}: dip X={xd:.2f} (corr {cs[i_min]:+.3f})  "
              f"hump X={xh:.2f} (corr {cs[i_max]:+.3f})  "
              f"zero X={xm:.2f}")
    comp = os.path.join(out, "composer")
    os.makedirs(comp, exist_ok=True)
    build_relay_vx(cast, os.path.join(comp, "relay-chosen-dips.wav"), chosen_dips)
    build_relay_vx(cast, os.path.join(comp, "relay-chosen-humps.wav"), chosen_humps)
    build_relay_vx(cast, os.path.join(comp, "relay-chosen-zero.wav"), chosen_mid)
    # persist the choices for the analyzer
    import json
    with open(os.path.join(comp, "choices.json"), "w") as f:
        json.dump({"dips": chosen_dips, "humps": chosen_humps,
                   "zero": chosen_mid}, f, indent=1)
    print("composer relays done.")


if __name__ == "__main__":
    main()
