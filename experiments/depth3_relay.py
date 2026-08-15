#!/usr/bin/env python3
"""Session 66: the relay of relays of relays — depth 3.

The named frontier from S65: does the transmission tax asymptote?
Layer 1 paid full price on raw voice edges (X=1.0: 0.735, X=2.0: 0.547).
Layer 2 paid only the rounding tax (X=1.0: 0.948, X=2.0: 0.891).
Layer 3 should pay a fraction of that fraction — approaching 1 from below,
or plateauing at the asymptotic tax.  Also: is fairness still a fixed
point at depth 3 (entry order -> ending delta ~0.5 dB)?

Also builds the depth-2 RESONANCE SWEEP: X swept 0 -> 5.5 s in 0.25 steps
on the layer-2 cast (durations 58.3-63.4 s, largest difference 5.10 s).
If the resonance period tracks the largest cast duration difference
(depth 1: 0.522 s = norman - lessac), the depth-2 teeth should sit at
multiples of ~5.1 s — the ceremony resonating with the outer cast's
internal clock.  Test of the resonance-mechanism prediction.

Usage: python3 depth3_relay.py <session65_dir> <out_root>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay, build_staircase


def main():
    s65 = sys.argv[1] if len(sys.argv) > 1 else "audio/session65"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session66"
    os.makedirs(out, exist_ok=True)

    l2 = os.path.join(s65, "layer2")
    # the layer-2 cast, ordered by ascending ceremony
    cast = [
        os.path.join(l2, "relay-of-relays-div-x0p3.wav"),
        os.path.join(l2, "relay-of-relays-div-x1p0.wav"),
        os.path.join(l2, "relay-of-relays-div-x2p0.wav"),
    ]
    # fourth voice: the convergent layer-2 (same material, opposite order)
    cast.append(os.path.join(l2, "relay-of-relays-conv-x1p0.wav"))

    # ---- 1. DEPTH 3: relay of relays of relays ----
    l3 = os.path.join(out, "layer3")
    os.makedirs(l3, exist_ok=True)
    for x in (0.3, 1.0, 2.0):
        label = f"x{str(x).replace('.', 'p')}"
        build_relay(cast, os.path.join(l3, f"roror-div-x{label}.wav"), x)

    # convergent depth-3: fairness-under-composition at depth 3
    cast_conv = [cast[3], cast[2], cast[1], cast[0]]
    build_relay(cast_conv, os.path.join(l3, "roror-conv-x1p0.wav"), 1.0)

    # staircase control at depth 3: the crowd made of chains of chains
    build_staircase(cast, os.path.join(l3, "staircase-of-relays-of-relays.wav"), 1.3)
    print(f"depth 3 done -> {l3}")

    # ---- 2. DEPTH-2 RESONANCE SWEEP (mechanism test) ----
    # sweep X 0 -> 5.5 step 0.25 on the layer-2 cast
    r2 = os.path.join(out, "resonance2")
    os.makedirs(r2, exist_ok=True)
    xs = [round(0.25 * i, 2) for i in range(0, 23)]  # 0 .. 5.50
    for x in xs:
        label = f"x{str(x).replace('.', 'p')}"
        build_relay(cast, os.path.join(r2, f"r2-{label}.wav"), x)
    print(f"depth-2 resonance sweep: {len(xs)} files -> {r2}")


if __name__ == "__main__":
    main()
