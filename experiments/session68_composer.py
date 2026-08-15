#!/usr/bin/env python3
"""Session 68, experiment 2: THE AIMING COMPOSER (composer-loop v2).

S67 made the analyzer a composer: chosen-X relays placed dips at handoffs
13-17 dB below body.  But the X was chosen at correlation extremes —
WHEREVER the material wanted.  v2 closes the loop the other way:

  1. PRE-SCREEN: per handoff, max |corr| over X — a strength grade.
     Weak pairs (S67's failed third hump, +0.199) are flagged before use.
  2. AIM: pick a TARGET time T.  A handoff dip's center sits at
     C_i = anchor_i - X_i/2  (anchor_i = end of voice i given earlier Xs),
     so X_i = 2 (anchor_i - T) places it — if X lands in [0.3, 5.5]
     (the reachable set is quantized by the ceremony's structure).
  3. CLOSE THE LOOP: compose -> measure the actual dip center in the
     50 ms envelope -> correct X_i += 2*error -> recompose.  3 rounds.
  4. Report the error series, achieved depth vs body, and the corr at
     the chosen X (the depth the material was willing to give there).

Also builds the DEPTH-FIRST control: X at each pair's corr minimum
(maximum depth, wherever it lands) — the trade-off frontier:
   aimed  = position exact, depth as given
   deep   = depth maximal, position as given

Usage: python3 session68_composer.py <session66_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay_vx
import analyze_conservation as AC

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
XMIN, XMAX = 0.30, 5.50


def env_db(path, winsize=0.05):
    data, sr = AC.read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12), winsize


def pair_corr(tail_env, head_env, X, ws=0.05):
    n = int(X / ws)
    if n < 5:
        return 0.0
    ta = tail_env[-n:] - tail_env[-n:].mean()
    hb = head_env[:n] - head_env[:n].mean()
    denom = np.sqrt((ta ** 2).sum() * (hb ** 2).sum())
    return float((ta * hb).sum() / denom) if denom > 0 else 0.0


def dur(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return float(o.stdout.strip())


def anchors(durs, xs):
    """anchor_i = start_i + dur_i; starts cascade through earlier Xs."""
    starts = [0.0]
    for i in range(len(durs) - 1):
        starts.append(starts[i] + durs[i] - xs[i])
    return [starts[i] + durs[i] for i in range(len(durs))], starts


def solve_xs(durs, targets, default=1.0):
    """targets: {handoff_i: T}. Solve X_i = 2(anchor_i - T) in order."""
    xs = [default] * (len(durs) - 1)
    for i in sorted(targets):
        a, _ = anchors(durs, xs)
        x = 2.0 * (a[i] - targets[i])
        xs[i] = min(XMAX, max(XMIN, x))
    return xs


def measure_dip(path, t_lo, t_hi, body_range):
    pd, ws = env_db(path)
    lo, hi = int(t_lo / ws), int(t_hi / ws)
    seg = pd[lo:hi]
    i_min = int(np.argmin(seg))
    center = (lo + i_min + 0.5) * ws
    body = pd[int(body_range[0] / ws):int(body_range[1] / ws)].mean()
    depth = float(body - seg[i_min])
    return center, float(depth), body


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session68/composer2"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [dur(v) for v in cast]
    total_default = sum(durs) - 3 * 1.0

    # ---------- 1. PRE-SCREEN ----------
    print("=" * 72)
    print("PRE-SCREEN (per handoff: max |corr|, argmax, corr minimum)")
    print("=" * 72)
    envs = [env_db(v)[0] for v in cast]
    Xs_scan = [round(0.05 * i, 2) for i in range(6, 111)]  # 0.30 .. 5.50
    curves = []
    for i, (a, b) in enumerate(zip(envs, envs[1:])):
        c = [(X, pair_corr(a, b, X)) for X in Xs_scan]
        curves.append(c)
        cs = np.array([x[1] for x in c])
        xs_ = np.array([x[0] for x in c])
        print(f"handoff {i}: max|corr| {np.abs(cs).max():.3f} at "
              f"X={xs_[np.argmax(np.abs(cs))]:.2f}   "
              f"corr min {cs.min():+.3f} at X={xs_[np.argmin(cs)]:.2f}   "
              f"[{'STRONG' if np.abs(cs).max() >= 0.5 else 'weak'}]")

    # ---------- 2-3. THE AIMING LOOP ----------
    print()
    print("=" * 72)
    print("THE AIMING LOOP (2 targets: handoffs 0 and 2; 3 rounds each)")
    print("=" * 72)
    # reachable check with default xs
    a0, _ = anchors(durs, [1.0, 1.0, 1.0])
    print(f"anchors (default X=1): "
          f"{[round(a, 2) for a in a0]}  total ~{total_default:.1f}s")
    T0 = round(a0[0] - 1.60, 2)   # ask the dip 1.6 s earlier than natural
    # handoff 2 anchor shifts with X0/X1; solve after X0 fixed
    report = {"targets": {}, "prescreen": []}
    for i, c in enumerate(curves):
        cs = np.array([x[1] for x in c])
        xs_ = [x[0] for x in c]
        report["prescreen"].append({
            "handoff": i, "max_abs_corr": float(np.abs(cs).max()),
            "corr_min": float(cs.min()),
            "x_at_corr_min": float(xs_[int(np.argmin(cs))])})

    # target 2: pick relative to anchor once X0 solved in round 0
    targets = {0: T0}
    xs = solve_xs(durs, targets)
    a_now, _ = anchors(durs, xs)
    T2 = round(a_now[2] - 2.20, 2)
    targets[2] = T2
    xs = solve_xs(durs, targets)
    print(f"targets: handoff0 -> T={T0}s   handoff2 -> T={T2}s")
    print(f"initial xs: {[round(x, 3) for x in xs]}")

    body_range = (30.0, total_default - 30.0)
    for rnd in range(3):
        f = os.path.join(out, f"aim-r{rnd}.wav")
        build_relay_vx(cast, f, list(xs))
        # measure both dips once
        meas = {}
        for i, T in sorted(targets.items()):
            c_meas, depth, body = measure_dip(
                f, T - 2.0, T + 2.0, body_range)
            meas[i] = (c_meas, depth)
            # corr at the chosen X (depth the material gives here)
            cs = dict(curves[i])
            print(f"  r{rnd} handoff {i}: X={xs[i]:.2f} "
                  f"corr(X)={cs.get(round(xs[i],2), float('nan')):+.3f}  "
                  f"dip center {c_meas:8.3f}s  target {T:8.3f}s  "
                  f"err {c_meas - T:+.3f}s  depth {depth:.1f} dB below body")
        # correct: X_i += 2 * (measured - target)
        for i, T in sorted(targets.items()):
            c_meas = meas[i][0]
            xs[i] = min(XMAX, max(XMIN, xs[i] + 2.0 * (c_meas - T)))
        report["targets"][f"round{rnd}"] = {
            "xs": [round(x, 3) for x in xs],
            "errors": [round(meas[i][0] - T, 3) for i, T in sorted(targets.items())]}

    # ---------- 4. DEPTH-FIRST CONTROL ----------
    print()
    print("=" * 72)
    print("DEPTH-FIRST CONTROL (X at corr minimum; position as given)")
    print("=" * 72)
    deep_xs = []
    for i, c in enumerate(curves):
        cs = [x[1] for x in c]
        deep_xs.append(c[int(np.argmin(cs))][0])
    f = os.path.join(out, "deep-first.wav")
    build_relay_vx(cast, f, deep_xs)
    for i, X in enumerate(deep_xs):
        a, _ = anchors(durs, deep_xs)
        center_pred = a[i] - X / 2.0
        print(f"  handoff {i}: X={X:.2f} -> predicted dip center "
              f"{center_pred:.2f}s (natural anchor {a0[i]:.2f}s, "
              f"shift {center_pred - a0[i]:+.2f}s)")

    with open(os.path.join(out, "aim-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'aim-report.json')}")


if __name__ == "__main__":
    main()
