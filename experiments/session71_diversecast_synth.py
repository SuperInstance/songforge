#!/usr/bin/env python3
"""Session 71, experiment 2 (part 1): THE DIVERSE LONG CAST.

S70's open question: does a DIVERSE crowd also hit the flat-topped wall,
or does the sign-flip (diverse casts OVERCOUNT themselves, +0.004..+0.200)
protect it?  The s64 cast couldn't answer — its voices are 4-5.5 s, too
short to assemble (duty ~0).  The roror cast (4 near-identical renditions
of one phrase, 243 s each) hit the wall at N*=156.

This script synthesizes THE SAME SCORE (session71_score.txt) with all
FIVE distinct piper voices (lessac, norman, joe, amy, aryah) — a diverse
cast of genuinely different speakers reading identical material, long
enough (~240 s) to fully assemble.  The score is held constant; only the
cast changes.  If the wall appears at the same N, the wall is a container
law.  If the clip fraction stays low, the sign-flip protects the crowd.

Usage: python3 session71_diversecast_synth.py <out_dir>
"""
import os
import subprocess
import sys

VOICES = {
    "lessac": "en_US-lessac-medium.onnx",
    "norman": "en_US-norman-medium.onnx",
    "joe": "en_US-joe-medium.onnx",
    "amy": "en_US-amy-medium.onnx",
    "aryah": "en_US-aryah-medium.onnx",
}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "audio/session71/diversecast"
    os.makedirs(out, exist_ok=True)
    score = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "session71_score.txt")).read().strip()
    vdir = os.path.expanduser("~/.local/share/piper-voices")
    for name, model in VOICES.items():
        path = os.path.join(out, f"voice-{name}.wav")
        if os.path.exists(path):
            print(f"{name}: exists, skip")
            continue
        p = subprocess.run(
            ["piper", "--model", os.path.join(vdir, model),
             "--output_file", path],
            input=score.encode(), capture_output=True)
        if p.returncode != 0:
            print(f"{name}: FAILED {p.stderr.decode()[-200:]}")
            continue
        d = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], capture_output=True, text=True)
        print(f"{name}: {float(d.stdout.strip()):.1f} s")


if __name__ == "__main__":
    main()
