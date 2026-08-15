#!/usr/bin/env python3
"""Build relay rounds: voices hand off the line via crossfade.

Two forms:
  - STAIRCASE: voices enter at fixed interval, all play at full volume to
    their natural end (the S62 drifting round). Density accumulates then
    depletes. This is the control.
  - RELAY: each voice's window ends where the next begins, joined by an
    equal-power crossfade of duration X. Density should stay ~constant
    (one voice-equivalent at all times) — the conservation-of-signal test.

Usage: python3 build_relay.py <voice_dir> <out_dir> [--interval 1.3] [--x 1.0]
Voices in voice_dir: lessac.wav norman.wav joe.wav amy.wav (mono wav, same sr).
"""
import argparse
import os
import subprocess
import sys


def probe_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(out.stdout.strip())


def ff(args):
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(f"ffmpeg failed: {' '.join(args)}\n{r.stderr[-2000:]}\n")
        sys.exit(1)


def build_staircase(voices, out, interval):
    """Control: all voices at full volume, entries at 0, i*interval, ..."""
    # pad each voice to the full length so amix keeps all tracks aligned
    parts = []
    for i, v in enumerate(voices):
        delay = int(i * interval * 1000)
        parts += ["-i", v]
    filt = []
    for i in range(len(voices)):
        delay = int(i * interval * 1000)
        filt.append(f"[{i}:a]adelay={delay}|{delay}[d{i}]")
    mix = "".join(f"[d{i}]" for i in range(len(voices)))
    filt.append(f"{mix}amix=inputs={len(voices)}:normalize=0[out]")
    ff(["ffmpeg", "-y"] + parts + ["-filter_complex", ";".join(filt),
        "-map", "[out]", out])


def build_relay(voices, out, x):
    """Voices hand off via equal-power crossfades of duration x.

    Voice i+1 starts x seconds before voice i ends. During the overlap,
    both are scaled by equal-power curves (cos/sin) so combined power
    stays constant. Outside overlaps each voice is alone at full gain.
    """
    durs = [probe_duration(v) for v in voices]
    # window starts: s0 = 0; s_{i+1} = s_i + dur_i - x
    starts = [0.0]
    for i in range(len(voices) - 1):
        starts.append(starts[i] + durs[i] - x)
    total = starts[-1] + durs[-1]
    sr = 22050  # piper default; probed below if needed
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", voices[0]],
        capture_output=True, text=True)
    sr = int(probe.stdout.strip())

    import numpy as np
    n = int(total * sr) + 1  # no slack; exact total duration
    mix = np.zeros(n, dtype=np.float64)

    def read(path):
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
             "-"], capture_output=True)
        return np.frombuffer(out.stdout, dtype=np.float32).astype(np.float64)

    # per-sample gain curves
    gains = []
    for i, v in enumerate(voices):
        g = np.ones(n, dtype=np.float64)
        s = int(starts[i] * sr)
        e = s + len(read(v))
        # fade in over first x seconds
        xi = int(x * sr)
        if xi > 0:
            g[s:s + xi] = np.sin(np.linspace(0, np.pi / 2, xi))
            g[s + xi: e - xi] = 1.0
            g[e - xi:e] = np.sin(np.linspace(np.pi / 2, 0, xi))
        gains.append(g)

    for i, v in enumerate(voices):
        data = read(v)
        s = int(starts[i] * sr)
        mix[s:s + len(data)] += data * gains[i][s:s + len(data)]

    # normalize to 0.98 peak to avoid clipping
    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix * (0.98 / peak)
    pcm = (mix * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i",
         "-", out], input=pcm.tobytes(), capture_output=True)
    print(f"relay: {out}  total={total:.2f}s  starts={[round(s,2) for s in starts]}")


def build_relay_vx(voices, out, xs):
    """Relay where handoff i uses crossfade duration xs[i] (chosen per pair).

    The composer version: xs come from the analyzer (envelope-correlation
    teeth per handoff), not from a uniform sweep.
    """
    assert len(xs) == len(voices) - 1, "one X per handoff"
    durs = [probe_duration(v) for v in voices]
    starts = [0.0]
    for i in range(len(voices) - 1):
        starts.append(starts[i] + durs[i] - xs[i])
    total = starts[-1] + durs[-1]
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", voices[0]],
        capture_output=True, text=True)
    sr = int(probe.stdout.strip())

    import numpy as np
    n = int(total * sr) + 1
    mix = np.zeros(n, dtype=np.float64)

    def read(path):
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1",
             "-"], capture_output=True)
        return np.frombuffer(out.stdout, dtype=np.float32).astype(np.float64)

    datas = [read(v) for v in voices]
    for i, data in enumerate(datas):
        s = int(starts[i] * sr)
        e = s + len(data)
        g = np.ones(e - s, dtype=np.float64)
        x_in = xs[i - 1] if i > 0 else 0.0      # fade-in from previous handoff
        x_out = xs[i] if i < len(voices) - 1 else 0.0  # fade-out to next
        xi = int(x_in * sr)
        xo = int(x_out * sr)
        if xi > 0:
            g[:xi] = np.sin(np.linspace(0, np.pi / 2, xi))
        if xo > 0:
            g[-xo:] = np.sin(np.linspace(np.pi / 2, 0, xo))
        mix[s:e] += data * g

    peak = np.abs(mix).max()
    if peak > 0:
        mix = mix * (0.98 / peak)
    pcm = (mix * 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", str(sr), "-ac", "1", "-i",
         "-", out], input=pcm.tobytes(), capture_output=True)
    print(f"relay-vx: {out}  total={total:.2f}s  xs={[round(x,2) for x in xs]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("voice_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--interval", type=float, default=1.3)
    ap.add_argument("--x", type=float, default=1.0)
    ap.add_argument("--order", default="lessac,norman,joe,amy")
    args = ap.parse_args()

    order = args.order.split(",")
    voices = [os.path.join(args.voice_dir, f"{v}.wav") for v in order]
    os.makedirs(args.out_dir, exist_ok=True)

    build_staircase(voices, os.path.join(args.out_dir, "relay-staircase.wav"),
                    args.interval)
    build_relay(voices, os.path.join(args.out_dir, "relay-x1.0.wav"), args.x)
    for x in (0.3, 0.5, 2.0):
        build_relay(voices, os.path.join(args.out_dir, f"relay-x{x}.wav"), x)


if __name__ == "__main__":
    main()
