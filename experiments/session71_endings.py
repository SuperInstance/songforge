#!/usr/bin/env python3
"""SongForge Session 71 — ending-signature profiler.

Analyzes the final 10 s of a track for the "solitary ending" signature:
- tail RMS vs body RMS (does it fade out or sustain?)
- tail spectral character: harmonic-to-noise-ish via spectral flatness
- tail centroid (bright = lone high voice?)
- vocal-band energy share in tail (200-3000 Hz)
- pitch stability: F0 estimate via autocorrelation over tail frames
"""
import sys, os, json, glob
import numpy as np
import soundfile as sf

def f0_autocorr(x, sr, fmin=80, fmax=1200):
    """Rough F0 via FFT-based autocorrelation on a downsampled slice."""
    if len(x) > 4 * sr:
        x = x[-int(4 * sr):]
    step = max(1, sr // 8000)
    x = x[::step]
    sr2 = sr / step
    n = len(x)
    lag_min = int(sr2 / fmax)
    lag_max = int(sr2 / fmin)
    # FFT autocorrelation
    x = x - x.mean()
    spec = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    ac = ac / (ac[0] + 1e-12)
    region = ac[lag_min:lag_max]
    if len(region) == 0:
        return 0.0
    peak = np.argmax(region)
    if region[peak] < 0.3:
        return 0.0
    return sr2 / (lag_min + peak)

def tail_profile(path, tail_s=10.0):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    n = len(data)
    dur = n / sr
    body = data[:int(n * 0.6)]  # first 60% = body
    tail = data[-int(tail_s * sr):] if n > int(tail_s * sr) else data

    def stats(x):
        rms = np.sqrt(np.mean(x ** 2) + 1e-12)
        fr = 2048
        win = np.hanning(fr)
        frames = np.array([x[j:j+fr] for j in range(0, max(1, len(x)-fr), 512)]) if len(x) > fr else np.zeros((1, fr))
        if len(frames) == 0:
            frames = np.zeros((1, fr))
        spec = np.abs(np.fft.rfft(frames * win, axis=1))
        ps = spec ** 2
        psum = ps.sum(axis=1) + 1e-12
        freqs = np.fft.rfftfreq(fr, 1 / sr)
        centroid = (ps * freqs).sum(axis=1) / psum
        flat = np.exp(np.mean(np.log(ps + 1e-12), axis=1)) / (np.mean(ps, axis=1) + 1e-12)
        vocal = ((freqs >= 200) & (freqs <= 3000)).sum()
        vshare = ps[:, :vocal].sum(axis=1) / psum
        return {
            "rms_db": 20 * np.log10(rms + 1e-12),
            "centroid": float(np.mean(centroid)),
            "flatness": float(np.mean(flat)),
            "vocal_share": float(np.mean(vshare)),
            "f0": f0_autocorr(x, sr),
        }

    b, t = stats(body), stats(tail)
    return {
        "file": os.path.basename(path),
        "duration_s": round(dur, 1),
        "body_rms_db": round(b["rms_db"], 2),
        "tail_rms_db": round(t["rms_db"], 2),
        "tail_drop_db": round(t["rms_db"] - b["rms_db"], 2),
        "body_centroid": round(b["centroid"], 0),
        "tail_centroid": round(t["centroid"], 0),
        "body_flatness": round(b["flatness"], 4),
        "tail_flatness": round(t["flatness"], 4),
        "body_vocal_share": round(b["vocal_share"], 3),
        "tail_vocal_share": round(t["vocal_share"], 3),
        "tail_f0_hz": round(float(t["f0"]), 1),
    }

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "audio/session71/grammar"
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "*.mp3"))):
        if "-spec" in p:
            continue
        try:
            rows.append(tail_profile(p))
        except Exception as e:
            print(f"ERR {p}: {e}", file=sys.stderr)
    for r in rows:
        print(f"{r['file']:22s} tail_drop={r['tail_drop_db']:+6.2f} dB  centroid {r['body_centroid']:.0f}->{r['tail_centroid']:.0f}  "
              f"vocal {r['body_vocal_share']:.2f}->{r['tail_vocal_share']:.2f}  flat {r['body_flatness']:.4f}->{r['tail_flatness']:.4f}  f0={r['tail_f0_hz']}")
    with open(os.path.join(d, "grammar-endings.json"), "w") as f:
        json.dump(rows, f, indent=1)

if __name__ == "__main__":
    main()
