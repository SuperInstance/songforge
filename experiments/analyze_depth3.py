#!/usr/bin/env python3
"""Session 66: the depth-3 tax series + resonance mechanism analysis.

TAX SERIES
----------
Relative tax (chain energy / crowd-of-same-material energy) by depth:
  depth 1 (S64): X=1.0: 0.735, X=2.0: 0.547
  depth 2 (S65): X=1.0: 0.948, X=2.0: 0.891
  depth 3 (this): X=1.0: ?, X=2.0: ?  -> does it asymptote?  To what?

RESONANCE SWEEPS
----------------
Two sweeps of X over the relay, both measured with the same windowed
profile (std of per-window power, gap fraction, energy):
  depth 1: morph-fine  X = 0.00..2.00 step 0.05 (41 files)
  depth 2: resonance2  X = 0.00..5.50 step 0.25 (23 files)
Resonance teeth = local maxima of profile std (alignment ringing) and
local minima of windowed energy (the dips where the tax rings).
If the resonance period is set by the largest cast duration difference
(depth 1: norman-lessac = 0.522 s -> teeth every ~0.5 s), depth 2's
cast (63.40, 61.30, 58.30, 61.30; largest diff 5.10 s) should ring at
X ~ 5.1 s multiples — a testable prediction of the mechanism.

Also: PAUSE-STRUCTURE PREDICTION.  Each voice has internal pauses
(piper sentence boundaries).  A handoff at X overlaps the tail of voice
i with the head of voice i+1.  If both contain internal silence at the
same offset, the overlap is "empty" -> energy dip / gap tooth.  Predict
the resonance teeth from the silence profile of each voice, compare to
the measured teeth.

Usage: python3 analyze_depth3.py <session64_dir> <session65_dir> <session66_dir>
"""
import os
import sys
import json
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_conservation as AC


def read_wav(path):
    return AC.read_wav(path)


def window_profile(path, winsize=0.25):
    """Per-window power (dB), silence fraction, windowed energy."""
    data, sr = read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    powers = (frames ** 2).mean(axis=1)
    pow_db = 10 * np.log10(powers + 1e-12)
    return {
        "n": n, "sr": sr, "win": winsize,
        "pow_db": pow_db,
        "sil_frac": float((pow_db < -45).mean()),
        "std": float(pow_db.std()),
        "energy": float((data ** 2).sum() / sr),
        "min": float(pow_db.min()),
    }


def find_teeth(xs, vals, order=1):
    """Local maxima of vals (order-1: peaks above both neighbors)."""
    teeth = []
    for i in range(order, len(xs) - order):
        if all(vals[i] > vals[j] for j in range(i - order, i + order + 1) if j != i):
            teeth.append((xs[i], vals[i]))
    return teeth


def find_dips(xs, vals, order=1):
    """Local minima of vals."""
    dips = []
    for i in range(order, len(xs) - order):
        if all(vals[i] < vals[j] for j in range(i - order, i + order + 1) if j != i):
            dips.append((xs[i], vals[i]))
    return dips


def sweep(dirname, pattern):
    files = sorted(f for f in os.listdir(dirname) if f.endswith(".wav")
                   and (pattern in f))
    rows = []
    for f in files:
        p = os.path.join(dirname, f)
        x = float(f.split("x")[1].replace("p", ".").replace(".wav", ""))
        prof = window_profile(p)
        prof["x"] = x
        rows.append(prof)
    rows.sort(key=lambda r: r["x"])
    return rows


def main():
    s64, s65, s66 = sys.argv[1:4]

    # ---------------- TAX SERIES ----------------
    print("=" * 72)
    print("TAX SERIES (chain energy / crowd-of-same-material energy)")
    print("=" * 72)
    # depth 3
    l3 = os.path.join(s66, "layer3")
    crowd3 = window_profile(os.path.join(l3, "staircase-of-relays-of-relays.wav"))
    for x in (0.3, 1.0, 2.0):
        f = os.path.join(l3, f"roror-div-xx{str(x).replace('.', 'p')}.wav")
        ch = window_profile(f)
        print(f"depth3 X={x:<4} chain {ch['energy']:.0f}  "
              f"crowd {crowd3['energy']:.0f}  tax {ch['energy']/crowd3['energy']:.4f}")
    # depth 2 (recompute from layer2 files, crowd = staircase-of-relays)
    l2 = os.path.join(s65, "layer2")
    crowd2 = window_profile(os.path.join(l2, "staircase-of-relays.wav"))
    for x in (0.3, 1.0, 2.0):
        f = os.path.join(l2, f"relay-of-relays-div-x{str(x).replace('.', 'p')}.wav")
        ch = window_profile(f)
        print(f"depth2 X={x:<4} chain {ch['energy']:.0f}  "
              f"crowd {crowd2['energy']:.0f}  tax {ch['energy']/crowd2['energy']:.4f}")
    # depth 1 (S64 relay vs S64 relay-staircase, same material)
    crowd1 = window_profile(os.path.join(s64, "divergent", "relay-staircase.wav"))
    for x in (0.3, 0.5, 1.0, 2.0):
        f = os.path.join(s64, "divergent", f"relay-x{x}.wav")
        ch = window_profile(f)
        print(f"depth1 X={x:<4} chain {ch['energy']:.0f}  "
              f"crowd {crowd1['energy']:.0f}  tax {ch['energy']/crowd1['energy']:.4f}")

    # ---------------- RESONANCE: DEPTH 1 (morph-fine) ----------------
    print()
    print("=" * 72)
    print("RESONANCE DEPTH 1 (morph-fine, X 0..2 step 0.05)")
    print("=" * 72)
    rows1 = sweep(os.path.join(s65, "morph-fine"), "fine-x")
    xs = [r["x"] for r in rows1]
    stds = [r["std"] for r in rows1]
    energy = [r["energy"] for r in rows1]
    sils = [r["sil_frac"] for r in rows1]
    t_std = find_teeth(xs, stds)
    d_ene = find_dips(xs, energy)
    t_sil = find_teeth(xs, sils)
    print(f"std teeth:   {[(round(x,2), round(v,2)) for x,v in t_std]}")
    print(f"energy dips: {[(round(x,2), round(v,1)) for x,v in d_ene]}")
    print(f"sil teeth:   {[(round(x,2), round(v,3)) for x,v in t_sil]}")
    # periodicity: autocorrelation of std vs X
    s = np.array(stds); s = s - s.mean()
    ac = np.correlate(s, s, "full")[len(s) - 1:]
    ac /= ac[0]
    # first significant secondary peak
    peaks = find_teeth(list(range(len(ac))), list(ac), order=2)
    lag_candidates = [(lag, v) for lag, v in peaks if lag > 2 and v > 0.15]
    print(f"std autocorr secondary peaks (lag samples, 0.05 s/sample): "
          f"{[(lag, round(v,2)) for lag,v in lag_candidates[:6]]}")

    # ---------------- RESONANCE: DEPTH 2 (resonance2) ----------------
    print()
    print("=" * 72)
    print("RESONANCE DEPTH 2 (resonance2, X 0..5.5 step 0.25)")
    print("=" * 72)
    rows2 = sweep(os.path.join(s66, "resonance2"), "r2-x")
    xs2 = [r["x"] for r in rows2]
    stds2 = [r["std"] for r in rows2]
    energy2 = [r["energy"] for r in rows2]
    sils2 = [r["sil_frac"] for r in rows2]
    t_std2 = find_teeth(xs2, stds2)
    d_ene2 = find_dips(xs2, energy2)
    t_sil2 = find_teeth(xs2, sils2)
    print(f"std teeth:   {[(round(x,2), round(v,2)) for x,v in t_std2]}")
    print(f"energy dips: {[(round(x,2), round(v,1)) for x,v in d_ene2]}")
    print(f"sil teeth:   {[(round(x,2), round(v,3)) for x,v in t_sil2]}")
    s2 = np.array(stds2); s2 = s2 - s2.mean()
    ac2 = np.correlate(s2, s2, "full")[len(s2) - 1:]
    ac2 /= ac2[0]
    peaks2 = find_teeth(list(range(len(ac2))), list(ac2), order=2)
    lag_candidates2 = [(lag, v) for lag, v in peaks2 if lag > 1 and v > 0.15]
    print(f"std autocorr secondary peaks (lag samples, 0.25 s/sample): "
          f"{[(lag, round(v,2)) for lag,v in lag_candidates2[:6]]}")

    # cast duration differences (predicted period)
    print()
    print("cast durations (layer-2): div-x0p3=63.40 div-x1p0=61.30 "
          "div-x2p0=58.30 conv-x1p0=61.30")
    durs = [63.40, 61.30, 58.30, 61.30]
    diffs = sorted({round(abs(a - b), 2) for a in durs for b in durs if a != b})
    print(f"duration differences: {diffs}  largest = {max(diffs)}")

    # ---------------- PAUSE-STRUCTURE PREDICTION ----------------
    print()
    print("=" * 72)
    print("PAUSE-STRUCTURE PREDICTION (depth 1: raw voices)")
    print("=" * 72)
    voices = ["lessac", "norman", "joe", "amy"]
    vpath = {v: os.path.join(s64, f"{v}.wav") for v in voices}
    # silence profile of each voice (250 ms windows, dB)
    profs = {v: window_profile(p) for v, p in vpath.items()}
    for v in voices:
        pr = profs[v]
        sil = pr["pow_db"] < -45
        # pauses: runs of silent windows, with start times
        runs = []
        start = None
        for i, s in enumerate(sil):
            if s and start is None:
                start = i * pr["win"]
            elif not s and start is not None:
                runs.append((start, i * pr["win"]))
                start = None
        if start is not None:
            runs.append((start, len(sil) * pr["win"]))
        print(f"{v:<8} dur {len(pr['pow_db'])*pr['win']:>6.2f}s  "
              f"pauses: {[(round(a,2), round(b-a,2)) for a,b in runs]}")

    # For each adjacent pair (i -> i+1) in relay order lessac->norman->joe->amy:
    # overlap at crossfade X covers tail of i (last X s) and head of i+1
    # (first X s).  Empty-overlap energy = sum of (silence in both).
    order = ["lessac", "norman", "joe", "amy"]
    print()
    print("predicted empty-overlap score vs X (mean over 3 handoffs)")
    Xs = [round(0.05 * i, 2) for i in range(41)]  # 0..2.0
    scores = []
    for X in Xs:
        parts = []
        for a, b in zip(order, order[1:]):
            pa = profs[a]["pow_db"]; pb = profs[b]["pow_db"]
            tail = pa[-(int(X / profs[a]["win"])):] if X > 0 else np.array([])
            head = pb[:int(X / profs[b]["win"])] if X > 0 else np.array([])
            # fraction of the overlap that is silent in BOTH voices
            n = min(len(tail), len(head))
            if n == 0:
                parts.append(0.0)
                continue
            both = (tail[-n:] < -45) & (head[:n] < -45)
            parts.append(both.mean())
        scores.append(np.mean(parts))
    t_pred = find_teeth(Xs, scores)
    print(f"predicted empty-overlap teeth (X, score): "
          f"{[(round(x,2), round(v,3)) for x,v in t_pred]}")
    print()
    print("measured depth-1 sil teeth:", [(round(x,2), round(v,3)) for x,v in t_sil])
    print("measured depth-1 std teeth:", [(round(x,2), round(v,2)) for x,v in t_std])


if __name__ == "__main__":
    main()
