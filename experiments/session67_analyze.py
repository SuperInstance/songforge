#!/usr/bin/env python3
"""Session 67 analysis: depth-4 tax, crowd ceiling, fairness, the frozen
clock, and the composer test (were the dips placed where predicted?).

Usage: python3 session67_analyze.py <s64> <s65> <s66> <s67>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_conservation as AC


def profile(path, winsize=0.25):
    data, sr = AC.read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    pow_db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    return {
        "dur": len(data) / sr,
        "energy": float((data ** 2).sum() / sr),
        "std": float(pow_db.std()),
        "pow_db": pow_db,
        "win": winsize,
    }


def tail_body(prof, tail_s=10.0):
    pd = prof["pow_db"]
    k = int(tail_s / prof["win"])
    return float(pd[-k:].mean() - pd[:-k].mean())


def teeth(xs, vals):
    out = []
    for i in range(2, len(xs) - 2):
        if vals[i] > vals[i - 1] and vals[i] > vals[i + 1] and \
           vals[i] > vals[i - 2] and vals[i] > vals[i + 2]:
            out.append((xs[i], vals[i]))
    return out


def dips(xs, vals):
    out = []
    for i in range(2, len(xs) - 2):
        if vals[i] < vals[i - 1] and vals[i] < vals[i + 1] and \
           vals[i] < vals[i - 2] and vals[i] < vals[i + 2]:
            out.append((xs[i], vals[i]))
    return out


def main():
    s64, s65, s66, s67 = sys.argv[1:5]

    # ================= 1. TAX SERIES + CROWD =================
    print("=" * 72)
    print("DEPTH-4 TABLE (tax = chain energy / crowd-of-same-material)")
    print("=" * 72)
    l4 = os.path.join(s67, "layer4")
    crowd4 = profile(os.path.join(l4, "staircase-of-everything.wav"))
    print(f"crowd4 (staircase-of-everything): dur {crowd4['dur']:.2f}s "
          f"energy {crowd4['energy']:.3e} std {crowd4['std']:.2f}")
    tax4 = {}
    for x in (0.3, 1.0, 2.0):
        f = os.path.join(l4, f"rororor-div-x{str(x).replace('.', 'p')}.wav")
        ch = profile(f)
        tax4[x] = ch["energy"] / crowd4["energy"]
        print(f"depth4 X={x:<4} chain {ch['energy']:.3e}  "
              f"tax {tax4[x]:.4f}")

    # voice-equivalents of the depth-4 crowd (veq = crowd energy / single
    # voice-equivalent reference).  S64 crowd=2.02, S65=3.14, S66=3.40
    # reference: energy of one layer-3 voice (the material of cast4)
    ref = profile(os.path.join(s66, "layer3", "roror-div-xx1p0.wav"))
    veq4 = crowd4["energy"] / ref["energy"]
    print(f"crowd4 veq (vs layer-3 voice): {veq4:.2f}   "
          f"[series: 2.02 -> 3.14 -> 3.40 -> {veq4:.2f}]")

    # deficit series
    print()
    print("DEFICIT (1 - tax):  X=1.0: 0.265 -> 0.052 -> 0.002 -> "
          f"{1 - tax4[1.0]:.4f}")
    print(f"                     X=2.0: 0.453 -> 0.109 -> 0.013 -> "
          f"{1 - tax4[2.0]:.4f}")
    f41 = (1 - tax4[1.0]) / 0.002 if 0.002 else float("nan")
    f42 = (1 - tax4[2.0]) / 0.013
    print(f"depth-3 -> depth-4 compounding factor: "
          f"X=1.0: {f41:.2f}   X=2.0: {f42:.2f}   (predicted ~0.25)")

    # ================= 2. FAIRNESS AT DEPTH 4 =================
    print()
    print("=" * 72)
    print("FAIRNESS AT DEPTH 4 (tail - body, dB)")
    print("=" * 72)
    div = profile(os.path.join(l4, "rororor-div-x1p0.wav"))
    conv = profile(os.path.join(l4, "rororor-conv-x1p0.wav"))
    td, tc = tail_body(div), tail_body(conv)
    print(f"divergent tail-body: {td:+.2f} dB   convergent: {tc:+.2f} dB   "
          f"delta {abs(td - tc):.2f} dB")
    print("[series: 0.50 (d1) -> 0.56 (d2) -> 0.53 (d3) -> "
          f"{abs(td - tc):.2f} (d4)]")

    # ================= 3. THE FROZEN CLOCK =================
    print()
    print("=" * 72)
    print("FROZEN CLOCK (depth-4 targeted sweep; depth-2 teeth were "
          "std 3.0/5.25, dips 2.5/3.0/4.5/5.0)")
    print("=" * 72)
    fc = os.path.join(s67, "frozen")
    rows = []
    for f in sorted(os.listdir(fc)):
        x = float(f.split("x")[1].replace("p", ".").replace(".wav", ""))
        rows.append((x, profile(os.path.join(fc, f))))
    rows.sort()
    xs = [r[0] for r in rows]
    print(f"{'X':>6} {'energy':>12} {'std':>6}")
    for x, p in rows:
        print(f"{x:>6.2f} {p['energy']:>12.3e} {p['std']:>6.2f}")
    t = teeth(xs, [p["std"] for _, p in rows])
    d = dips(xs, [p["energy"] for _, p in rows])
    print(f"std teeth (coarse, 7 pts): {[(round(a,2), round(b,2)) for a,b in t]}")
    print(f"energy dips (coarse):      {[(round(a,2), round(b,1)) for a,b in d]}")
    print("verdict: eyeball — do minima sit at 2.5-3.25 / 5.0-5.25 "
          "and maxima at 4.0?")

    # ================= 4. THE COMPOSER TEST =================
    print()
    print("=" * 72)
    print("COMPOSER TEST (chosen-X relays: were dips placed?)")
    print("=" * 72)
    comp = os.path.join(s67, "composer")
    choices = json.load(open(os.path.join(comp, "choices.json")))

    # handoff windows: overlap region of handoff i = [start_{i+1},
    # start_i + dur_i].  Recompute starts per relay from file durations.
    import subprocess

    def dur(p):
        o = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", p],
                           capture_output=True, text=True)
        return float(o.stdout.strip())

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in
            ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
             "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")]
    durs = [dur(v) for v in cast]

    def handoff_energy(path, xs):
        p = profile(path)
        pd, w = p["pow_db"], p["win"]
        starts = [0.0]
        for i in range(3):
            starts.append(starts[i] + durs[i] - xs[i])
        out = []
        for i in range(3):
            a = int(starts[i + 1] / w)
            b = int((starts[i] + durs[i]) / w)
            out.append(float(pd[a:b].mean()))
        body = float(pd.mean())
        return out, body

    for name, key in (("chosen-dips", "dips"), ("chosen-humps", "humps"),
                      ("chosen-zero", "zero")):
        xs = choices[key]
        ho, body = handoff_energy(
            os.path.join(comp, f"relay-{name}.wav"), xs)
        rel = [h - body for h in ho]
        print(f"{name:<14} X={xs}  handoff-body (dB): "
              f"{[round(r, 2) for r in rel]}  overall std "
              f"{profile(os.path.join(comp, f'relay-{name}.wav'))['std']:.2f}")
    print()
    print("prediction: chosen-dips handoffs sit BELOW body; "
          "chosen-humps ABOVE; chosen-zero in between.")


if __name__ == "__main__":
    main()
