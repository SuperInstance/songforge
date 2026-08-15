#!/usr/bin/env python3
"""Session 66: resonance period FFT + envelope-correlation prediction.

The silence-granularity prediction failed (piper voices have almost no
internal pauses at 250 ms).  Two refinements:

1. PERIOD FFT: dominant period of the std-vs-X curve at depth 1
   (morph-fine, 0.05 s steps) and depth 2 (resonance2, 0.25 s steps).
   S65 claimed 0.5 s period at depth 1 (norman-lessac = 0.522 s).
   The fine sweep shows teeth at 0.25 spacing in the first cluster —
   is the true period 0.25 (half the largest duration difference)?

2. ENVELOPE CORRELATION: instead of hard silence, correlate the energy
   envelope of voice i's tail with voice i+1's head as X varies.  Where
   the envelopes line up (constructive), energy is high; where they
   anti-align, dips.  Teeth of the correlation curve = predicted dips.
   Compare to measured energy dips.

Usage: python3 refine_resonance.py <session64_dir> <session65_dir> <session66_dir>
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_conservation as AC


def read_wav(path):
    return AC.read_wav(path)


def env_db(path, winsize=0.05):
    """Fine energy envelope (dB), 50 ms windows."""
    data, sr = read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)


def pow_db_profile(path, winsize=0.25):
    data, sr = read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    return 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)


def fft_period(xs, vals, spacing):
    """Dominant period of a sampled curve via FFT (returns seconds)."""
    v = np.array(vals)
    v = v - v.mean()
    n = len(v)
    spec = np.abs(np.fft.rfft(v * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, spacing)
    # skip DC and the longest half
    keep = freqs > 1.0 / (xs[-1] - xs[0] + spacing)
    if not keep.any():
        return None, None
    spec = spec.copy()
    spec[~keep] = 0
    # exclude very high freq noise: period > 2*spacing
    keep2 = freqs < 1.0 / (2 * spacing)
    spec[~keep2] = 0
    i = int(np.argmax(spec))
    return freqs[i], 1.0 / freqs[i] if freqs[i] > 0 else None


def main():
    s64, s65, s66 = sys.argv[1:4]

    # ---------- DEPTH 1 PERIOD ----------
    mf = os.path.join(s65, "morph-fine")
    files = sorted(f for f in os.listdir(mf) if f.endswith(".wav"))
    rows = []
    for f in files:
        x = float(f.split("x")[1].replace("p", ".").replace(".wav", ""))
        p = os.path.join(mf, f)
        data, sr = read_wav(p)
        win = int(0.25 * sr)
        n = len(data) // win
        frames = data[: n * win].reshape(n, win)
        std = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12).std()
        rows.append((x, std))
    rows.sort()
    xs = [r[0] for r in rows]; stds = [r[1] for r in rows]
    f1, p1 = fft_period(xs, stds, 0.05)
    print(f"depth1 std-vs-X: dominant freq {f1:.3f} Hz -> period {p1:.3f} s")
    print(f"  (norman-lessac = 0.522 s; half = 0.261 s; teeth at 0.25 spacing)")

    # ---------- DEPTH 2 PERIOD ----------
    r2 = os.path.join(s66, "resonance2")
    files2 = sorted(f for f in os.listdir(r2) if f.endswith(".wav"))
    rows2 = []
    for f in files2:
        x = float(f.split("x")[1].replace("p", ".").replace(".wav", ""))
        p = os.path.join(r2, f)
        data, sr = read_wav(p)
        win = int(0.25 * sr)
        n = len(data) // win
        frames = data[: n * win].reshape(n, win)
        std = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12).std()
        rows2.append((x, std))
    rows2.sort()
    xs2 = [r[0] for r in rows2]; stds2 = [r[1] for r in rows2]
    f2, p2 = fft_period(xs2, stds2, 0.25)
    print(f"depth2 std-vs-X: dominant freq {f2:.3f} Hz -> period {p2:.3f} s")
    print(f"  (layer-2 cast diffs: 2.1, 3.0, 5.1; largest 5.1; half = 2.55)")
    print(f"  measured teeth: 3.0, 5.25 (std); dips 2.5, 3.0, 4.5, 5.0")

    # ---------- ENVELOPE CORRELATION PREDICTION (depth 1) ----------
    print()
    print("ENVELOPE-CORRELATION PREDICTION (depth 1, X 0..2 step 0.05)")
    voices = ["lessac", "norman", "joe", "amy"]
    envs = {v: env_db(os.path.join(s64, f"{v}.wav")) for v in voices}
    Xs = [round(0.05 * i, 2) for i in range(41)]
    # overlap energy: energy in last X s of voice i + first X s of voice i+1
    # (the region the crossfade covers), as a function of X.
    # dips = X where the overlap region is quiet in both voices.
    def overlap_energy(X):
        tot = 0.0
        for a, b in zip(voices, voices[1:]):
            ea, eb = envs[a], envs[b]
            n = int(X / 0.05)
            if n == 0:
                continue
            tail = ea[-n:]
            head = eb[:n]
            tot += float(np.mean(tail) + np.mean(head))
        return tot / 3.0

    oe = [overlap_energy(X) for X in Xs]
    dips = []
    for i in range(2, len(Xs) - 2):
        if oe[i] < oe[i-1] and oe[i] < oe[i+1] and oe[i] < oe[i-2] and oe[i] < oe[i+2]:
            dips.append((Xs[i], round(oe[i], 2)))
    print(f"predicted overlap-energy dips: {dips}")
    print(f"measured energy dips: [(0.95,0.2),(1.05,0.2),(1.15,0.2),(1.35,0.2),(1.8,0.1)]")

    # normalized cross-correlation of tail/head envelopes at each X
    print()
    print("tail-head envelope correlation at each X (depth 1)")
    corrs = []
    for X in Xs:
        cs = []
        for a, b in zip(voices, voices[1:]):
            ea, eb = envs[a], envs[b]
            n = int(X / 0.05)
            if n < 3:
                cs.append(0.0)
                continue
            ta = ea[-n:] - ea[-n:].mean()
            hb = eb[:n] - eb[:n].mean()
            denom = np.sqrt((ta**2).sum() * (hb**2).sum())
            cs.append(float((ta * hb).sum() / denom) if denom > 0 else 0.0)
        corrs.append(np.mean(cs))
    cteeth = []
    for i in range(2, len(Xs) - 2):
        if corrs[i] > corrs[i-1] and corrs[i] > corrs[i+1]:
            cteeth.append((Xs[i], round(corrs[i], 3)))
    print(f"correlation teeth: {cteeth}")


if __name__ == "__main__":
    main()
