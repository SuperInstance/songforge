#!/usr/bin/env python3
"""Quantify round disassembly: silence floor, gap structure vs stagger interval.
Usage: python3 analyze_gaps.py <dir> <glob>
"""
import sys
import glob
import numpy as np
import analyze_rms as A

THRESH = -30.0  # dB: below this = "silence" between voices

def gap_stats(path, win=0.25):
    x, sr = A.read_wav(path)
    dur = len(x) / sr
    n = int(dur / win)
    floors = []
    for i in range(n):
        seg = x[int(i * win * sr):int((i + 1) * win * sr)]
        floors.append(A.rms_db(seg))
    floors = np.array(floors)
    silent = floors < THRESH
    # count silence runs (gaps)
    gaps = 0
    in_gap = False
    for s in silent:
        if s and not in_gap:
            gaps += 1
            in_gap = True
        elif not s:
            in_gap = False
    min_db = floors.min()
    mean_db = floors.mean()
    return min_db, gaps, dur

if __name__ == "__main__":
    pattern = sys.argv[1]
    files = sorted(glob.glob(pattern))
    print(f"{'file':<32} {'dur':>6} {'floor':>7} {'mean':>7} {'gaps':>5}")
    for f in files:
        min_db, gaps, dur = gap_stats(f)
        x, sr = A.read_wav(f)
        mean_db = A.rms_db(x)
        print(f"{f.split('/')[-1]:<32} {dur:>6.2f} {min_db:>7.1f} {mean_db:>7.1f} {gaps:>5}")
