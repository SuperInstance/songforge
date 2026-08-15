#!/usr/bin/env python3
"""Session 70, experiment 3c: THE SIGN HONESTY TABLE.

Assembles the v5 sign-reading results into one table: for each handoff,
the prescreen sign at the rented width vs the placed feature sign and
magnitude.  The verdict: the prescreen predicts the placed sign ONLY at
the tight width (X=0.25, 3/3) and fails at wide widths (X=1.0, 0/3).

Law refinement (clearance law v5): X rents the address and the depth.
The SIGN is a LOCAL property of the transition — readable before the
build only when the prescreen is itself local (tight X).  At wide X the
prescreen dilutes the transition with body material and reads GHOST
SIGNS — the correlation of the bodies, not the seam.  The tight width
is the only honest width in all three currencies: depth, address, sign.

Usage: python3 session70_composer5c.py <composer5_dir> <composer5b_dir>
"""
import json
import os
import sys

GRID = (0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.50)


def main():
    d5 = sys.argv[1] if len(sys.argv) > 1 else "audio/session70/composer5"
    d5b = sys.argv[2] if len(sys.argv) > 2 else "audio/session70/composer5b"
    r5 = json.load(open(os.path.join(d5, "composer5-report.json")))
    r5b = json.load(open(os.path.join(d5b, "composer5b-report.json")))

    print("=" * 78)
    print("THE SIGN HONESTY TABLE (prescreen vs placed, by rented width)")
    print("=" * 78)
    print(f"{'h':>2} {'X':>5} {'corr':>7} {'read':>5} {'placed':>6} "
          f"{'depth':>6} {'bump':>6} {'match':>6}")
    rows = {}
    for i in range(3):
        for X, tag in ((0.25, "tight"), (1.0, "wide")):
            src = r5 if X == 0.25 else r5b
            key = f"h{i}"
            if X == 1.0:
                e = r5b[f"h{i}"]
                corr = e["corr"]
            else:
                e = r5[f"h{i}"]
                corr = e["corr_grid"][str(X)]
            read = "BUMP" if corr > 0 else "DIP"
            placed = e["placed"]
            match = read == placed
            rows[f"{i}-{X}"] = {"corr": corr, "read": read,
                                "placed": placed, "match": match,
                                "depth": e["depth"], "bump": e["bump"]}
            print(f"{i:2d} {X:5.2f} {corr:+7.3f} {read:>5} {placed:>6} "
                  f"{e['depth']:6.1f} {e['bump']:6.1f} "
                  f"{'MATCH' if match else 'no':>6}")

    tight = sum(1 for k, v in rows.items() if k.endswith("0.25") and v["match"])
    wide = sum(1 for k, v in rows.items() if k.endswith("1.0") and v["match"])
    print()
    print(f"tight (X=0.25): {tight}/3 prescreen matches placed sign")
    print(f"wide  (X=1.00): {wide}/3 prescreen matches placed sign")
    print()
    print("VERDICT: the sign is readable before the build only at the")
    print("tight width.  At wide X the prescreen reads body correlation,")
    print("not the seam — ghost signs.  The tight width is the only")
    print("honest width in all three currencies: depth, address, sign.")

    with open(os.path.join(d5, "sign-honesty.json"), "w") as fj:
        json.dump({"rows": rows, "tight_matches": tight,
                   "wide_matches": wide}, fj, indent=1)
    print(f"\ntable -> {os.path.join(d5, 'sign-honesty.json')}")


if __name__ == "__main__":
    main()
