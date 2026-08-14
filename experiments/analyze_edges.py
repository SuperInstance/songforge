#!/usr/bin/env python3
"""Measure first/last window RMS for entry/exit analysis.
Usage: python3 analyze_edges.py <file.wav> [--window 1.5]
"""
import sys
import numpy as np
import analyze_rms as A

def main():
    path = sys.argv[1]
    win = 1.5
    if "--window" in sys.argv:
        win = float(sys.argv[sys.argv.index("--window") + 1])
    x, sr = A.read_wav(path)
    dur = len(x) / sr
    first = x[:int(win * sr)]
    last = x[-int(win * sr):]
    # middle 1s (the "body") for reference
    mid = x[int(len(x)/2):int(len(x)/2) + sr]
    print(f"{path}  ({dur:.2f}s)")
    print(f"  first {win}s: {A.rms_db(first):.1f} dB   mid 1s: {A.rms_db(mid):.1f} dB   last {win}s: {A.rms_db(last):.1f} dB")
    # asymmetry: how different is the ending from the body
    print(f"  ending-vs-body: {A.rms_db(last) - A.rms_db(mid):+.1f} dB   opening-vs-body: {A.rms_db(first) - A.rms_db(mid):+.1f} dB")

if __name__ == "__main__":
    main()
