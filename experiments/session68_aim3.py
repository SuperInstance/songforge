#!/usr/bin/env python3
"""Session 68, experiment 3: THE DAMPED AIMER — composer-loop v3.

v2 (session68_composer.py) findings that v3 must fix:
  A. THE LOOP RINGS.  err series handoff0 +0.685 -> -1.615 -> -0.465,
     handoff2 +0.905 -> -0.195 -> +1.105.  Gain-2 correction overshoots;
     X clamps at the 5.5 wall.  Fix: gain 0.5, no clamp oscillation,
     report convergence rate.
  B. THE AIMER CHASES GHOSTS.  Depth readings alternate 72.8/73.3 dB
     (inherited fade-rim silences, S67 finding) with 14.9-18.0 dB
     (placed dips).  The ±2 s argmin window catches the inherited rim,
     not the placed dip.  Fix: measure the LOCAL MIN in a band around
     the PREDICTED dip center (anchor - X/2) — the placed seam — and
     report the ghost (global min) separately.
  C. THE PRE-SCREEN SATURATES AT ITS OWN FLOOR.  max|corr| and corr-min
     both landed at X = 0.30, the scan start.  Fix: scan 0.05 -> 5.50
     (S67 found its dips at X = 0.25, below v2's floor).

Also builds THE MEDIUM'S CEILING CURVE: same layer-3 cast, N = 2/4/8,
interval 1.3, each in s16 AND f32 — the container tax (s16/f32 energy)
as a function of the census.  S67: N=4 relay tax 0.12-0.38%; S68 census
N=8: 0.9739 (2.6% clipped).  Is the medium's share monotone in N?

Usage: python3 session68_aim3.py <session66_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay_vx, probe_duration
import analyze_conservation as AC

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
XMIN, XMAX = 0.05, 5.50


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


def anchors(durs, xs):
    starts = [0.0]
    for i in range(len(durs) - 1):
        starts.append(starts[i] + durs[i] - xs[i])
    return [starts[i] + durs[i] for i in range(len(durs))], starts


def solve_xs(durs, targets, default=1.0):
    xs = [default] * (len(durs) - 1)
    for i in sorted(targets):
        a, _ = anchors(durs, xs)
        x = 2.0 * (a[i] - targets[i])
        xs[i] = min(XMAX, max(XMIN, x))
    return xs


def local_min(pd, ws, center, half=0.45):
    """Min envelope in [center-half, center+half]; returns (t, val)."""
    lo = max(0, int((center - half) / ws))
    hi = min(len(pd), int((center + half) / ws) + 1)
    if hi - lo < 2:
        return center, -np.inf
    seg = pd[lo:hi]
    i = int(np.argmin(seg))
    return (lo + i + 0.5) * ws, float(seg[i])


def body_level(pd, ws, t_lo, t_hi):
    lo, hi = int(t_lo / ws), int(t_hi / ws)
    return float(pd[max(0, lo):hi].mean())


def staircase_pair(voices, out_s16, out_f32, interval, sr):
    """Same census in s16 and f32 — measures the container tax directly."""
    import numpy as np
    from session68_census import read_f32, write_wav  # reuse S68 helpers

    datas = [read_f32(v) for v in voices]
    end = max(int(len(d) / sr + i * interval) for i, d in enumerate(datas))
    n = (end + 1) * sr
    mix = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(datas):
        s = int(i * interval * sr)
        mix[s:s + len(d)] += d
    write_wav(mix, sr, out_s16, "s16le")
    write_wav(mix, sr, out_f32, "f32le")


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session68/aim3"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    durs = [probe_duration(v) for v in cast]
    envs = [env_db(v)[0] for v in cast]
    ws = 0.05
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())

    # ---------- C. PRESCREEN, TRUE FLOOR (0.05 not 0.30) ----------
    print("=" * 72)
    print("PRE-SCREEN v3 (scan 0.05 .. 5.50 — below v2's 0.30 floor)")
    print("=" * 72)
    Xs_scan = [round(0.05 * i, 2) for i in range(1, 111)]
    curves = []
    for i, (a, b) in enumerate(zip(envs, envs[1:])):
        c = [(X, pair_corr(a, b, X)) for X in Xs_scan]
        curves.append(c)
        cs = np.array([x[1] for x in c])
        xs_ = np.array([x[0] for x in c])
        print(f"handoff {i}: max|corr| {np.abs(cs).max():.3f} at "
              f"X={xs_[np.argmax(np.abs(cs))]:.2f}   corr min "
              f"{cs.min():+.3f} at X={xs_[np.argmin(cs)]:.2f}   "
              f"[{'STRONG' if np.abs(cs).max() >= 0.5 else 'weak'}]")

    # ---------- A+B. DAMPED AIMING LOOP, GHOST-PROOF ----------
    print()
    print("=" * 72)
    print("THE DAMPED AIMER (gain 0.5; measure at predicted center,")
    print("                  report the ghost separately; 4 rounds)")
    print("=" * 72)
    a0, _ = anchors(durs, [1.0] * 3)
    T0 = round(a0[0] - 1.60, 2)
    targets = {0: T0}
    xs = solve_xs(durs, targets)
    a_now, _ = anchors(durs, xs)
    T2 = round(a_now[2] - 2.20, 2)
    targets[2] = T2
    xs = solve_xs(durs, targets)
    print(f"targets: handoff0 -> T={T0}s   handoff2 -> T={T2}s")
    print(f"initial xs: {[round(x, 3) for x in xs]}")

    total_default = sum(durs) - 3 * 1.0
    body = (30.0, total_default - 30.0)
    report = {"targets": {}, "prescreen": [], "ghosts": {}}
    for i, c in enumerate(curves):
        cs = np.array([x[1] for x in c])
        report["prescreen"].append({
            "handoff": i, "max_abs_corr": float(np.abs(cs).max()),
            "corr_min": float(cs.min()),
            "x_at_corr_min": float(Xs_scan[int(np.argmin(cs))])})

    for rnd in range(4):
        f = os.path.join(out, f"damp-r{rnd}.wav")
        build_relay_vx(cast, f, list(xs))
        pd, ws_ = env_db(f)
        b = body_level(pd, ws_, *body)
        line = []
        for i, T in sorted(targets.items()):
            a, _ = anchors(durs, xs)
            pred = a[i] - xs[i] / 2.0          # where the seam SHOULD sit
            t_place, v_place = local_min(pd, ws_, pred)
            t_ghost, v_ghost = local_min(pd, ws_, T, half=2.0)
            depth_placed = b - v_place
            depth_ghost = b - v_ghost
            err = t_place - T
            line.append(f"  r{rnd} h{i}: X={xs[i]:.2f} placed@"
                        f"{t_place:7.3f}s (pred {pred:7.3f}) "
                        f"err {err:+.3f} depth {depth_placed:5.1f} dB | "
                        f"ghost {depth_ghost:5.1f} dB @{t_ghost:7.3f}")
            print(line[-1])
            report["ghosts"].setdefault(rnd, {})[f"h{i}"] = {
                "x": round(xs[i], 3), "placed_err": round(err, 3),
                "placed_depth": round(depth_placed, 1),
                "ghost_depth": round(depth_ghost, 1)}
        # damped correction: X += 0.5 * (placed - target), clamped
        for i, T in sorted(targets.items()):
            a, _ = anchors(durs, xs)
            pred = a[i] - xs[i] / 2.0
            t_place, _ = local_min(pd, ws_, pred)
            xs[i] = min(XMAX, max(XMIN, xs[i] + 0.5 * (t_place - T)))
        report["targets"][f"round{rnd}"] = {"xs": [round(x, 3) for x in xs]}

    # ---------- THE MEDIUM'S CEILING CURVE ----------
    print()
    print("=" * 72)
    print("THE MEDIUM'S CEILING CURVE (same cast, N=2/4/8, s16 vs f32)")
    print("=" * 72)
    ref_en = AC.analyze(cast[1], 1.0)["energy"] if False else None
    import analyze_conservation as AC2
    ref = AC2.analyze(cast[1], 1.0)
    ref_energy = ref["total_energy"]
    for N in (2, 4, 8):
        voices = (cast * 2)[:N]
        base = os.path.join(out, f"tax-N{N}")
        staircase_pair(voices, base + "-s16.wav", base + "-f32.wav", 1.3, sr)
        e16 = AC2.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC2.analyze(base + "-f32.wav", 1.0)["total_energy"]
        tax = e16 / e32
        print(f"N={N}: veq(s16) {e16 / ref_energy:6.3f}   "
              f"veq(f32) {e32 / ref_energy:6.3f}   "
              f"container tax {tax:.4f}   N/veq32 "
              f"{N / (e32 / ref_energy):.3f}")
        report[f"tax-N{N}"] = {"veq_s16": float(e16 / ref_energy),
                               "veq_f32": float(e32 / ref_energy),
                               "container_tax": float(tax)}

    with open(os.path.join(out, "aim3-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'aim3-report.json')}")


if __name__ == "__main__":
    main()
