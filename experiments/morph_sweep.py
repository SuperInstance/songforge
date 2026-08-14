#!/usr/bin/env python3
"""Session 65: the morph sweep and the relay of relays.

MORPH SWEEP
-----------
Crossfade X swept 0.0 -> 2.0 s in 0.1 steps on the divergent quartet.
The staircase is the crowd (accumulation, veq ~2).  X=0 is the hard-cut
chain (instantaneous handoffs, one voice at a time, zero overlap).
X >= 0.5 is the seamless chain.  The sweep maps the transition: where
do the handoff gaps close, and how does the transmission tax (energy
lost to fades) scale with ceremony length?

RELAY OF RELAYS (chains of chains)
----------------------------------
The four divergent relay outputs (x0.3, x0.5, x1.0, x2.0) become the
four voices of a second-generation relay.  Tests whether conservation
is closed under composition: is the chain of chains still a chain?
  - layer2 divergent X in {0.3, 1.0, 2.0}: tax compounding across layers
  - layer2 convergent X=1.0: does the fairness result (entry-order
    collapse to 0.5 dB) survive composition?  Is fairness a fixed point?
  - layer2 staircase control: the crowd made of chains.

Usage: python3 morph_sweep.py <session64_dir> <out_root>
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_relay import build_relay, build_staircase


def main():
    s64 = sys.argv[1] if len(sys.argv) > 1 else "audio/session64"
    out = sys.argv[2] if len(sys.argv) > 2 else "audio/session65"
    os.makedirs(out, exist_ok=True)

    div = os.path.join(s64, "divergent")
    conv = os.path.join(s64, "convergent")

    # ---- 1. MORPH SWEEP: X = 0.0 .. 2.0 step 0.1 (divergent order) ----
    morph = os.path.join(out, "morph")
    os.makedirs(morph, exist_ok=True)
    xs = [round(0.1 * i, 1) for i in range(0, 21)]
    voices = [os.path.join(s64, f"{v}.wav") for v in
              ("lessac", "norman", "joe", "amy")]
    for x in xs:
        label = f"x{str(x).replace('.', 'p')}"
        build_relay(voices, os.path.join(morph, f"morph-{label}.wav"), x)
    print(f"morph sweep: {len(xs)} files -> {morph}")

    # ---- 2. RELAY OF RELAYS (layer 2) ----
    l2 = os.path.join(out, "layer2")
    os.makedirs(l2, exist_ok=True)

    # layer-2 voices are the layer-1 relay outputs, ordered by ascending X
    # (ascending ceremony) for the divergent chain-of-chains
    for x in (0.3, 1.0, 2.0):
        voices = [os.path.join(div, f"relay-x{v}.wav") for v in (0.3, 0.5, 1.0, 2.0)]
        build_relay(voices, os.path.join(l2, f"relay-of-relays-div-x{str(x).replace('.', 'p')}.wav"), x)

    # convergent layer-2: fairness-under-composition test
    voices_c = [os.path.join(conv, f"relay-x{v}.wav") for v in (0.3, 0.5, 1.0, 2.0)]
    build_relay(voices_c, os.path.join(l2, "relay-of-relays-conv-x1p0.wav"), 1.0)

    # staircase control: the crowd made of chains (divergent, interval 1.3)
    build_staircase([os.path.join(div, f"relay-x{v}.wav") for v in (0.3, 0.5, 1.0, 2.0)],
                    os.path.join(l2, "staircase-of-relays.wav"), 1.3)
    print(f"layer 2 done -> {l2}")


if __name__ == "__main__":
    main()
