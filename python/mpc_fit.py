# Fit Multi-ProbCut regressions from the CSV written by the mpc_stats
# example (versor-eval/examples/mpc_stats.rs).
# For each phase and depth pair (deep d, shallow d') in (4,2), (6,4),
# (8,6), fits v_d ~ a * v_d' + b and the residual standard deviation.
# Output: text parameter file, one line per (phase, d, dprime):
#   phase d dprime a b sigma
# consumed by versor-search (implementation follows Buro's
# Multi-ProbCut papers; no GPL source was consulted).

import os
import sys

import numpy as np

PAIRS = [(4, 2), (6, 4), (8, 6)]
COL = {2: 1, 4: 2, 6: 3, 8: 4}

def main():
    if len(sys.argv) != 3:
        sys.exit("usage: mpc_fit.py stats.csv params.txt")
    if not os.path.exists(sys.argv[1]):
        sys.exit(f"{sys.argv[1]}: not found (the mpc_stats example writes it, see models/README.md)")
    data = np.loadtxt(sys.argv[1], delimiter=",")
    lines = []
    for phase in range(2, 14):
        rows = data[data[:, 0] == phase]
        for d, dp in PAIRS:
            x = rows[:, COL[dp]]
            y = rows[:, COL[d]]
            a, b = np.polyfit(x, y, 1)
            resid = y - (a * x + b)
            sigma = float(resid.std())
            lines.append(f"{phase} {d} {dp} {a:.6f} {b:.6f} {sigma:.6f}\n")
            print(
                f"phase {phase} pair ({d},{dp}): a={a:.3f} b={b:+.3f} "
                f"sigma={sigma:.3f} n={len(rows)}"
            )
    with open(sys.argv[2], "w") as f:
        f.write("# phase d dprime a b sigma\n")
        f.writelines(lines)

if __name__ == "__main__":
    main()
