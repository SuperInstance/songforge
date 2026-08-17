#!/usr/bin/env python3
"""SongForge Session 71 — time-structure profiler.

For each track: split into 16 time segments; report per-segment RMS dB,
centroid Hz, and onset density. Answers: does the track have structure
(intro/build/chorus/outro arcs) or is it ambient drift? Also computes a
"structure score": the RMS time-series' autocorrelation at the beat/bar
scale vs pure randomness, and an intro/outro arc description.
"""
import sys, os, json, glob
import numpy as np
import soundfile as sf

def profile(path, nseg=16):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    seg = n // nseg
    out = []
    fr = 2048
    win = np.hanning(fr)
    freqs = np.fft.rfftfreq(fr, 1 / sr)
    for i in range(nseg):
        chunk = data[i*seg:(i+1)*seg]
        rms = np.sqrt(np.mean(chunk ** 2) + 1e-12)
        frames = np.array([chunk[j:j+fr] for j in range(0, len(chunk)-fr, 512)]) if len(chunk) > fr else np.zeros((1, fr))
        if len(frames) == 0:
            frames = np.zeros((1, fr))
        spec = np.abs(np.fft.rfft(frames * win, axis=1))
        ps = spec ** 2
        psum = ps.sum(axis=1) + 1e-12
        centroid = (ps * freqs).sum(axis=1) / psum
        out.append({
            "seg": i,
            "t_s": round(i * seg / sr, 1),
            "rms_db": round(20 * np.log10(rms + 1e-12), 2),
            "centroid_hz": round(float(np.mean(centroid)), 0),
        })
    return out

def structure_score(rms_db_series):
    """Autocorrelation of the segment RMS series at lags 1..4 — how much
    periodic structure (vs noise) the loudness arc has."""
    x = np.array(rms_db_series)
    x = x - x.mean()
    if np.std(x) == 0:
        return 0.0
    ac = np.correlate(x, x, mode="full")[len(x)-1:]
    ac = ac / ac[0]
    return round(float(np.mean(ac[1:5])), 3)

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "audio/session71/grammar"
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.mp3"))):
        if "-spec" in p:
            continue
        prof = profile(p)
        rms = [r["rms_db"] for r in prof]
        cent = [r["centroid_hz"] for r in prof]
        ss = structure_score(rms)
        rows.append({
            "file": os.path.basename(p),
            "structure_score": ss,
            "rms_arc": rms,
            "centroid_arc": cent,
            "rms_range_db": round(max(rms) - min(rms), 2),
            "first_half_centroid": round(float(np.mean(cent[:8])), 0),
            "second_half_centroid": round(float(np.mean(cent[8:])), 0),
        })
    for r in rows:
        print(f"{r['file']:22s} struct={r['structure_score']:+.3f} rms_range={r['rms_range_db']:5.2f} "
              f"centroid {r['first_half_centroid']:.0f}->{r['second_half_centroid']:.0f} Hz")
        print(f"    rms: {r['rms_arc']}")
    with open(os.path.join(d, "grammar-structure.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nWrote grammar-structure.json ({len(rows)} tracks)")

if __name__ == "__main__":
    main()
