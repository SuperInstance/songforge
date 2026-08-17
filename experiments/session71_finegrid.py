#!/usr/bin/env python3
"""Session 71, experiment 1: THE PHASE-TRANSITION FINE GRID.

S70 found the tax curve BREAKS at the wall: pre-wall fit tax ~ N^-0.061
vs post-wall N^-0.419 (exponent ratio 0.14) — a phase transition, not a
smooth power law — with the break somewhere in N=96..160 (the conditional
clip crossed 50% at N* = 156.37, interpolated between 128 and 160).

This experiment builds the FINE GRID between 96 and 160 (N = 96, 104,
112, 120, 128, 136, 144, 152, 160 — 104..152 are new builds; 96/128/160
already exist from S70) and then:
  A. Re-derives the 50% crossing N* by interpolation on the fine grid.
  B. SLIDING-WINDOW EXPONENT FIT: fit tax ~ a*N^-b on a sliding window
     of 4 consecutive points; locate the break N_b where b jumps, and
     measure the jump's sharpness (pre/post exponent ratio).
  C. CONDITIONAL TAX BREAK: same sliding fit on the census-window tax
     (tax_in from the conditional report) — does the break sharpen or
     smear when the window is honest?
  D. RAIL-ENERGY LEDGER: rail_share vs N on the fine grid — is the
     wall a reservoir (rail share accelerates past N_b)?

Usage: python3 session71_finegrid.py <session66_dir> <taxwall_dir> <out_dir>
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from session68_census import read_f32
import analyze_conservation as AC
from session70_taxwall import staircase_metrics

L3 = ("roror-div-xx0p3.wav", "roror-div-xx1p0.wav",
      "roror-div-xx2p0.wav", "roror-conv-x1p0.wav")
NEW_NS = (104, 112, 120, 136, 144, 152)   # 96/128/160 already built
INTERVAL = 1.3
SR = 22050


def build_missing(s66, taxwall):
    """Build the fine-grid N values not already present."""
    l3 = os.path.join(s66, "layer3")
    cast = [os.path.join(l3, n) for n in L3]
    sr_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate", "-of", "csv=p=0", cast[0]],
        capture_output=True, text=True)
    sr = int(sr_probe.stdout.strip())
    rows = json.load(open(os.path.join(taxwall, "taxwall-report.json")))
    ref_energy = AC.analyze(cast[1], 1.0)["total_energy"]
    for N in NEW_NS:
        base = os.path.join(taxwall, f"taxwall-N{N:03d}")
        if os.path.exists(base + "-s16.wav") and os.path.exists(base + "-f32.wav"):
            print(f"N={N}: exists, skip")
            continue
        voices = (cast * (N // len(cast) + 1))[:N]
        _, clip_frac, flat_frac, rail_e = staircase_metrics(
            voices, base + "-s16.wav", base + "-f32.wav", INTERVAL, sr)
        e16 = AC.analyze(base + "-s16.wav", 1.0)["total_energy"]
        e32 = AC.analyze(base + "-f32.wav", 1.0)["total_energy"]
        rows[str(N)] = {"veq_s16": e16 / ref_energy, "veq_f32": e32 / ref_energy,
                        "tax": float(e16 / e32), "clip_frac": clip_frac,
                        "flat_frac": flat_frac,
                        "wall_ratio": float(flat_frac / clip_frac)
                        if clip_frac > 0 else 0.0,
                        "rail_share": float(rail_e / e16) if e16 > 0 else 0.0,
                        "rms_db_s16": AC.analyze(base + "-s16.wav", 1.0)["rms_db"]}
        print(f"N={N:3d}: tax {rows[str(N)]['tax']:.4f}  "
              f"clip {clip_frac*100:5.2f}%  rail {rows[str(N)]['rail_share']*100:5.2f}%")
    merged = {}
    for k, v in rows.items():
        try:
            merged[int(k)] = v
        except (ValueError, TypeError):
            merged[k] = v
    with open(os.path.join(taxwall, "taxwall-report.json"), "w") as fj:
        json.dump(merged, fj, indent=1, default=str)


def conditional_fine(taxwall, out):
    """Census-window metrics on the fine grid (mirrors taxwall2)."""
    rows = json.load(open(os.path.join(taxwall, "taxwall-report.json")))
    NS = tuple(sorted(int(k) for k in rows if str(k).isdigit()))
    dur0 = 241.288
    cond = json.load(open(os.path.join(taxwall,
                                       "taxwall-conditional-report.json")))
    for N in NS:
        if str(N) in cond and cond[str(N)].get("clip_in") is not None:
            continue
        f32 = read_f32(os.path.join(taxwall, f"taxwall-N{N:03d}-f32.wav"))
        s16r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i",
             os.path.join(taxwall, f"taxwall-N{N:03d}-s16.wav"),
             "-f", "f32le", "-ac", "1", "-"], capture_output=True)
        s16 = np.frombuffer(s16r.stdout, dtype=np.float32).astype(np.float64)
        win_lo = max(0.0, (N - 1) * INTERVAL)
        win_hi = min(dur0, len(f32) / SR)
        if win_hi <= win_lo:
            cond[str(N)] = {"win": [win_lo, win_hi], "clip_in": None,
                            "tax_in": 0.0, "win_share": 0.0}
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
              f"tax_in {tax_in:.4f}")
    with open(os.path.join(taxwall, "taxwall-conditional-report.json"),
              "w") as fj:
        json.dump(cond, fj, indent=1, default=str)


def sliding_exponent(NS, vals, win=4):
    """Sliding-window power-law fit: b_k from NS[k:k+win], vals[k:k+win]."""
    out = []
    for k in range(len(NS) - win + 1):
        n = np.array(NS[k:k + win], dtype=float)
        v = np.array(vals[k:k + win], dtype=float)
        mask = v > 0
        if mask.sum() < win - 1:
            continue
        b, a = np.polyfit(np.log(n[mask]), np.log(v[mask]), 1)
        out.append({"k": k, "N_mid": float(np.sqrt(n[0] * n[-1])),
                    "b": float(b), "a": float(np.exp(a)),
                    "N_range": [int(n[0]), int(n[-1])]})
    return out


def main():
    s66 = sys.argv[1] if len(sys.argv) > 1 else "audio/session66"
    taxwall = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/taxwall"
    out = sys.argv[3] if len(sys.argv) > 3 else "audio/session71/finegrid"
    os.makedirs(out, exist_ok=True)

    build_missing(s66, taxwall)
    conditional_fine(taxwall, out)

    rows = json.load(open(os.path.join(taxwall, "taxwall-report.json")))
    NS = tuple(sorted(int(k) for k in rows if str(k).isdigit()))
    tax = [rows[str(n)]["tax"] for n in NS]
    clip = [rows[str(n)]["clip_frac"] for n in NS]
    rail = [rows[str(n)]["rail_share"] for n in NS]

    # A. 50% crossing on the fine grid (linear interpolation in log-N)
    cross50 = None
    for i in range(len(NS) - 1):
        if clip[i] < 0.5 <= clip[i + 1]:
            l0, l1 = np.log(NS[i]), np.log(NS[i + 1])
            c0, c1 = clip[i], clip[i + 1]
            cross50 = float(np.exp(l0 + (0.5 - c0) * (l1 - l0) / (c1 - c0)))
            break
    print("=" * 72)
    print(f"FINE GRID: N = {NS}")
    print(f"50% conditional crossing: N* = {cross50:.2f}  (S70: 156.37)")

    # B. sliding exponent on the GLOBAL tax curve
    se_global = sliding_exponent(NS, tax)
    print("\nSLIDING EXPONENT (global tax, window=4):")
    for r in se_global:
        print(f"  N {r['N_range'][0]:3d}-{r['N_range'][1]:3d}  "
              f"b = {r['b']:+.4f}")
    b_pre = [r["b"] for r in se_global if r["N_range"][0] <= 64]
    b_post = [r["b"] for r in se_global if r["N_range"][0] >= 96]
    print(f"  pre-wall b ~ {np.mean(b_pre):+.4f}  post-wall b ~ "
          f"{np.mean(b_post):+.4f}  ratio {np.mean(b_pre)/np.mean(b_post):.3f}")

    # C. sliding exponent on the CONDITIONAL tax
    cond = json.load(open(os.path.join(taxwall,
                                       "taxwall-conditional-report.json")))
    NS_c = [n for n in NS if str(n) in cond and
            cond[str(n)].get("tax_in") not in (None, 0.0)]
    tax_c = [cond[str(n)]["tax_in"] for n in NS_c]
    se_cond = sliding_exponent(NS_c, tax_c)
    print("\nSLIDING EXPONENT (conditional tax, window=4):")
    for r in se_cond:
        print(f"  N {r['N_range'][0]:3d}-{r['N_range'][1]:3d}  "
              f"b = {r['b']:+.4f}")
    if len(se_cond) >= 3:
        b_pre_c = [r["b"] for r in se_cond if r["N_range"][0] <= 64]
        b_post_c = [r["b"] for r in se_cond if r["N_range"][0] >= 96]
        print(f"  pre-wall b ~ {np.mean(b_pre_c):+.4f}  post-wall b ~ "
              f"{np.mean(b_post_c):+.4f}  ratio "
              f"{np.mean(b_pre_c)/np.mean(b_post_c):.3f}")

    # D. RAIL-ENERGY LEDGER: rail_share vs N
    print("\nRAIL-ENERGY LEDGER (rail_share = pinned energy / s16 energy):")
    for n, r in zip(NS, rail):
        print(f"  N={n:3d}  rail_share {r*100:6.2f}%")
    # logistic fit: rail_share ~ L / (1 + exp(-k (N - N_half)))
    rp = np.array(rail, dtype=float)
    nn = np.array(NS, dtype=float)
    L = float(rp.max()) if rp.max() > 0 else 1.0
    mask = rp > 0
    if mask.sum() >= 5:
        y = np.clip(rp[mask] / L, 1e-9, 1 - 1e-9)
        z = np.log(y / (1 - y))
        k, b0 = np.polyfit(nn[mask], z, 1)
        N_half = float(-b0 / k)
        print(f"  logistic: L={L:.4f}  k={k:+.4f}  N_half={N_half:.1f}  "
              f"(wall at half-reservoir)")
        # power-law tail: rail_share ~ a*N^c for N >= 96
        tail = nn[nn >= 96]
        rt = rp[nn >= 96]
        if len(tail) >= 3 and (rt > 0).all():
            c, a = np.polyfit(np.log(tail), np.log(rt), 1)
            print(f"  tail power law: rail_share ~ {np.exp(a):.4f} * N^{c:+.3f}")

    report = {"NS": list(NS), "cross50": cross50,
              "sliding_global": se_global, "sliding_cond": se_cond,
              "rail_share": {str(n): round(r, 6) for n, r in zip(NS, rail)},
              "tax": {str(n): round(t, 6) for n, t in zip(NS, tax)}}
    with open(os.path.join(out, "finegrid-report.json"), "w") as fj:
        json.dump(report, fj, indent=1)
    print(f"\nreport -> {os.path.join(out, 'finegrid-report.json')}")


if __name__ == "__main__":
    main()
