#!/usr/bin/env python3
"""Session 68, experiment 1: THE CENSUS LAW — does veq -> N?

S67 refuted the crowd ceiling: veq series 2.02 -> 3.14 -> 3.40 -> 3.99 was
bounded by the CAST SIZE (N=4 at every depth), not by architecture.  The
law: togetherness is a census — veq -> N as material/interval -> inf.

Test: a staircase of N=8 (the layer-3 cast twice).  Predictions:
  - veq8 ~= 8 (census holds when N doubles at fixed depth)
  - interval 5.0 lowers veq by the entry-ramp fraction only
  - entry-order fate (fwd vs rev tail-body, 10% proportional window)
    washes out like the relay fate did
  - the s16 container may clip an 8-voice sum: a float32 no-clip twin
    measures the CONTAINER'S ceiling separately from the cast's.

Usage: python3 session68_census.py <session66_dir> <out_dir>
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_staircase, probe_duration
import analyze_conservation as AC

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")


def read_f32(path):
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-"],
        capture_output=True)
    return np.frombuffer(out.stdout, dtype=np.float32).astype(np.float64)


def write_wav(mix, sr, path, fmt="s16le"):
    if fmt == "s16le":
        pcm = (np.clip(mix, -1.0, 1.0) * 32767).astype(np.int16)
        arg = ["-f", "s16le"]
    else:
        pcm = mix.astype(np.float32)
        arg = ["-f", "f32le"]
    subprocess.run(
        ["ffmpeg", "-y"] + arg + ["-ar", str(sr), "-ac", "1", "-i", "-",
         "-c:a", ("pcm_s16le" if fmt == "s16le" else "pcm_f32le"), path],
        input=pcm.tobytes(), capture_output=True)


def staircase_np(voices, out, interval, clip=True):
    """numpy staircase: sum at full gain, entries at i*interval.

    clip=True  -> s16 wav like the house amix builder (clips at 1.0)
    clip=False -> float32 wav, the true unclipped census
    """
    datas = [read_f32(v) for v in voices]
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", voices[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())
    end = max(int(len(d) / sr + i * interval) for i, d in enumerate(datas))
    n = (end + 1) * sr
    mix = np.zeros(n, dtype=np.float64)
    for i, d in enumerate(datas):
        s = int(i * interval * sr)
        mix[s:s + len(d)] += d
    if clip:
        # match house style: peak-normalize ONLY if above 1.0 is clipped
        write_wav(mix, sr, out, "s16le")
    else:
        write_wav(mix, sr, out, "f32le")
    total = n / sr
    print(f"staircase np ({'s16' if clip else 'f32'}): {out}  "
          f"total={total:.2f}s  N={len(voices)}  interval={interval}")


def profile(path, winsize=0.25):
    data, sr = AC.read_wav(path)
    win = int(winsize * sr)
    n = len(data) // win
    frames = data[: n * win].reshape(n, win)
    pow_db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    return {"dur": len(data) / sr,
            "energy": float((data ** 2).sum() / sr),
            "std": float(pow_db.std()),
            "pow_db": pow_db, "win": winsize}


def tail_body(prof, frac=0.10):
    pd = prof["pow_db"]
    k = max(1, int(len(pd) * frac))
    return float(pd[-k:].mean() - pd[:-k].mean())


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session68/census"
    os.makedirs(out, exist_ok=True)

    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]

    # ---- build: N=8 census, both orders, two intervals, plus no-clip twin
    fwd = cast + cast                       # A B C D A B C D
    rev = cast[::-1] + cast[::-1]           # D C B A D C B A
    build_staircase(fwd, os.path.join(out, "census8-fwd.wav"), 1.3)
    build_staircase(rev, os.path.join(out, "census8-rev.wav"), 1.3)
    build_staircase(fwd, os.path.join(out, "census8-int5.wav"), 5.0)
    staircase_np(fwd, os.path.join(out, "census8-f32.wav"), 1.3, clip=False)

    # ---- analysis
    print()
    print("=" * 72)
    print("THE CENSUS (veq = crowd energy / one layer-3 voice energy)")
    print("=" * 72)
    ref = profile(cast[1])
    rows = {}
    for name in ("census8-fwd", "census8-rev", "census8-int5", "census8-f32"):
        p = profile(os.path.join(out, f"{name}.wav"))
        rows[name] = p
        print(f"{name:<14} dur {p['dur']:8.2f}s  energy {p['energy']:.4e}  "
              f"veq {p['energy'] / ref['energy']:.3f}  std {p['std']:.2f}")
    print()
    print(f"series so far: N=1: 1.00 | 2.02 (N=2,d1) -> 3.14 (N=4,d2) -> "
          f"3.40 (N=4,d3) -> 3.99 (N=4,d4)")
    print(f"census N=8 (d3 material): fwd "
          f"{rows['census8-fwd']['energy'] / ref['energy']:.2f}   "
          f"no-clip f32 {rows['census8-f32']['energy'] / ref['energy']:.2f}   "
          f"interval-5 {rows['census8-int5']['energy'] / ref['energy']:.2f}")

    clip_ratio = (rows["census8-fwd"]["energy"] /
                  rows["census8-f32"]["energy"])
    print(f"container tax (s16 energy / f32 energy): {clip_ratio:.4f}  "
          f"(1.000 = no clipping; <1 = the medium took its share)")

    # ---- fairness at N=8 (entry order, proportional 10% window)
    print()
    print("=" * 72)
    print("FAIRNESS AT N=8 (tail - body, 10% proportional window)")
    print("=" * 72)
    tf = tail_body(rows["census8-fwd"])
    tr = tail_body(rows["census8-rev"])
    print(f"fwd {tf:+.2f} dB   rev {tr:+.2f} dB   delta {abs(tf - tr):.2f} dB")
    print("[relay fate series: 3.23 (d1) -> 0.73 (d2) -> 0.07 (d3) -> "
          "0.10 (d4)]")


if __name__ == "__main__":
    main()
