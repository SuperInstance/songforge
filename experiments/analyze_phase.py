#!/usr/bin/env python3
"""Phase diagram: silence fraction vs stagger interval (sweep threshold).
Usage: python3 analyze_phase.py <dir>
"""
import sys
import glob
import numpy as np
import analyze_rms as A

def silence_fraction(path, thresh_db, win=0.1):
    x, sr = A.read_wav(path)
    n = int(len(x) / sr / win)
    silent = 0
    for i in range(n):
        seg = x[int(i * win * sr):int((i + 1) * win * sr)]
        if A.rms_db(seg) < thresh_db:
            silent += 1
    return silent / max(n, 1)

if __name__ == "__main__":
    d = sys.argv[1]
    files = sorted(glob.glob(f"{d}/canon-iv-*.wav"))
    print("interval  dur    floor   sil@-30  sil@-35  sil@-40")
    for f in files:
        iv = f.split("iv-")[1].replace(".wav", "")
        x, sr = A.read_wav(f)
        dur = len(x) / sr
        min_db, _, _ = None, None, None
        floors = []
        win = 0.1
        n = int(dur / win)
        for i in range(n):
            seg = x[int(i * win * sr):int((i + 1) * win * sr)]
            floors.append(A.rms_db(seg))
        floor = min(floors)
        s30 = silence_fraction(f, -30)
        s35 = silence_fraction(f, -35)
        s40 = silence_fraction(f, -40)
        print(f"{iv:>8}  {dur:>6.2f}  {floor:>6.1f}  {s30:>7.3f}  {s35:>7.3f}  {s40:>7.3f}")
