#!/usr/bin/env python3
"""Session 71, experiment 2 (part 2): DOES A DIVERSE CROWD HIT THE WALL?

The decisive cast test.  The roror cast (4 near-identical renditions of
one phrase, 243 s each) hit the flat-topped wall at N* = 156 (conditional
clip 50.53% at N=160, cross50 = 156.37).  The s64 cast (4 distinct
speakers, 4-5.5 s utterances) never assembled — duty ~0, tax ~0.995 even
at N=32.  This experiment gives the diverse cast what the roror cast had:
LONG voices (312-352 s), fully assembling at 1.3 s interval, reading the
SAME score (session71_score.txt) — the material held constant, only the
cast changes.

Three builds per N (all at interval 1.3):
  A. RMS-MATCHED: every voice scaled to the roror reference RMS — the
     fair wall test (same loudness as the S70 wall series).
  B. ENERGY-NORMALIZED: every voice total energy = 1.0 — the honest
     dividend test (S70 censuscast2 method, now with assembling voices).
  C. RAW: untouched synthesis loudness — the loudness effect.

Measures per N (1..192, taxwall grid): tax (s16/f32), clip fraction,
flat fraction, rail share, veq (f32) vs the roror reference energy, and
the conditional (full-census-window) clip/tax.

Questions:
  - Does the diverse crowd hit the wall at all?  At what N*?
  - Does the diverse dividend stay POSITIVE as the crowd assembles
    (overcounts itself), where the identical roror cast went negative
    (-0.403 @ 32)?
  - Is the phase-transition break present in the diverse tax curve?

Usage: python3 session71_diversecast_wall.py <diversecast_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session68_census import read_f32, write_wav
import analyze_conservation as AC
from session70_taxwall import staircase_metrics

VOICES = ("voice-lessac.wav", "voice-norman.wav", "voice-joe.wav",
          "voice-amy.wav")
NS = (1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192)
INTERVAL = 1.3
SR = 22050


def rms(v):
    d = read_f32(v)
    return float(np.sqrt((d ** 2).mean())), d


def main():
    vdir = sys.argv[1] if len(sys.argv) > 1 else "audio/session71/diversecast"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session71/diversecast-wall"
    os.makedirs(out, exist_ok=True)

    # roror reference RMS (the S70 wall series reference voice)
    s66 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "audio/session66/layer3/roror-div-xx1p0.wav")
    ref_rms, _ = rms(s66)
    ref_energy = AC.analyze(s66, 1.0)["total_energy"]
    print(f"roror reference RMS = {ref_rms:.5f}, energy = {ref_energy:.3f}")

    raw = [read_f32(os.path.join(vdir, v)) for v in VOICES]
    rms_matched = []
    normed = []
    for i, d in enumerate(raw):
        r = float(np.sqrt((d ** 2).mean()))
        rms_matched.append(d * (ref_rms / r) if r > 0 else d)
        e = float((d ** 2).sum())
        normed.append(d * np.sqrt(1.0 / e) if e > 0 else d)
        print(f"{VOICES[i]}: raw rms {r:.5f} -> matched {ref_rms:.5f}")

    # overlap duty at N=192 (window survival check)
    end = max(int(len(d) / SR + i * INTERVAL)
              for i, d in enumerate(rms_matched * 48))
    n = int((end + 1) * SR)
    on = np.zeros(n, dtype=np.int8)
    for i in range(192):
        d = rms_matched[i % 4]
        s = int(i * INTERVAL * SR)
        on[s:s + len(d)] += 1
    duty = float((on >= 2).mean())
    win_lo = (192 - 1) * INTERVAL
    print(f"N=192: duty {duty*100:.1f}%  census window [{win_lo:.1f}, "
          f"{min(len(d)/SR for d in rms_matched):.1f}] "
          f"= {max(0, min(len(d)/SR for d in rms_matched)-win_lo):.1f} s")

    # write scaled casts to disk (staircase_metrics reads file paths)
    cast_files = {}
    for tag, cast in (("rms", rms_matched), ("norm", normed), ("raw", raw)):
        files = []
        for i, d in enumerate(cast):
            p = os.path.join(out, f"cast-{tag}-{VOICES[i]}")
            write_wav(d, SR, p, "f32le")
            files.append(p)
        cast_files[tag] = files

    rows = {}
    for tag in ("rms", "norm", "raw"):
        cast = cast_files[tag]
        sub = {}
        for N in NS:
            voices = (cast * (N // 4 + 1))[:N]
            base = os.path.join(out, f"{tag}-N{N:03d}")
            _, clip_frac, flat_frac, rail_e = staircase_metrics(
                voices, base + "-s16.wav", base + "-f32.wav", INTERVAL, SR)
            e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
            e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
            sub[str(N)] = {"veq_s16": e16 / ref_energy,
                           "veq_f32": e32 / ref_energy,
                           "tax": float(e16 / e32),
                           "clip_frac": float(clip_frac),
                           "flat_frac": float(flat_frac),
                           "wall_ratio": float(flat_frac / clip_frac)
                           if clip_frac > 0 else 0.0,
                           "rail_share": float(rail_e / e16) if e16 > 0 else 0.0}
            print(f"[{tag}] N={N:3d}: tax {sub[str(N)]['tax']:.4f}  "
                  f"clip {clip_frac*100:5.2f}%  flat {flat_frac*100:5.2f}%  "
                  f"veq32 {sub[str(N)]['veq_f32']:7.3f}")
        rows[tag] = sub

    # conditional (census-window) analysis on the RMS-matched series
    cond = {}
    dur0 = min(len(d) / SR for d in rms_matched)
    for N in NS:
        f32 = read_f32(os.path.join(out, f"rms-N{N:03d}-f32.wav"))
        s16r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i",
             os.path.join(out, f"rms-N{N:03d}-s16.wav"),
             "-f", "f32le", "-ac", "1", "-"], capture_output=True)
        s16 = np.frombuffer(s16r.stdout, dtype=np.float32).astype(np.float64)
        win_lo = max(0.0, (N - 1) * INTERVAL)
        win_hi = min(dur0, len(f32) / SR)
        if win_hi <= win_lo:
            cond[str(N)] = {"win": [win_lo, win_hi], "clip_in": None}
            print(f"[cond] N={N:3d}: census window VANISHED")
            continue
        a, b = int(win_lo * SR), int(win_hi * SR)
        seg32 = f32[a:b]
        seg16 = s16[a:b]
        clip_in = float((np.abs(seg32) >= 1.0).mean())
        tax_in = float((seg16 ** 2).sum() / (seg32 ** 2).sum()) \
            if (seg32 ** 2).sum() > 0 else 0.0
        cond[str(N)] = {"win": [win_lo, win_hi], "clip_in": clip_in,
                        "tax_in": tax_in, "win_share": (b - a) / len(f32)}
        print(f"[cond] N={N:3d}: clip_in {clip_in*100:5.2f}%  "
              f"tax_in {tax_in:.4f}  win {win_hi-win_lo:.1f}s")

    report = {"ref_rms": ref_rms, "ref_energy": ref_energy,
              "rows": rows, "conditional": cond}
    with open(os.path.join(out, "diversecast-wall-report.json"), "w") as fj:
        json.dump(report, fj, indent=1, default=str)
    print(f"\nreport -> {os.path.join(out, 'diversecast-wall-report.json')}")


if __name__ == "__main__":
    main()
