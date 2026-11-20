#!/usr/bin/env python3
"""Table 2: NOTA selection when it is correct (GR) and when it is not (DR),
NOTA discrimination D = GR - DR, and the last-option rate L on the original
items (0-shot, exhaustive design).

    python analysis/build_table2.py [--csv results/tables/cce_tables_final.csv] [--outdir .]

D is computed from the displayed (rounded) GR and DR, as in the paper. Bold D:
the 95% bootstrap CI of the per-item gap excludes 0. Also prints the median
and maximum CI half-width quoted in the caption. Writes rows2.tex.
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import DATASETS, DEFAULT_TABLES, MODELS, pct1, round1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_TABLES)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)

    def g(m, ds, setting, metric, col="value"):
        r = d[(d.model == m) & (d.dataset == ds) & (d.shots == 0)
              & (d.setting == setting) & (d.metric == metric)]
        assert len(r) == 1, (m, ds, metric)
        return float(r.iloc[0][col])

    lines, half = [], []
    for e in MODELS:
        if e is None:
            lines.append("\\addlinespace")
            continue
        name, m = e
        cells = []
        for ds in DATASETS:
            gr = g(m, ds, "exhaustive", "A_GR")
            dr = g(m, ds, "exhaustive", "NOTA_DR")
            lo = g(m, ds, "exhaustive", "NOTA_discrimination", "ci95_lo")
            hi = g(m, ds, "exhaustive", "NOTA_discrimination", "ci95_hi")
            half.append(100 * (hi - lo) / 2)
            last = g(m, ds, "orig", "last_letter_rate")
            disc = str(round1(100 * gr) - round1(100 * dr))
            disc = "0.0" if disc == "-0.0" else disc
            disc = f"\\textbf{{{disc}}}" if (lo > 0 or hi < 0) else disc
            cells += [pct1(gr), pct1(dr), disc, pct1(last)]
        lines.append(f"{name:<29}& " + " & ".join(cells) + " \\\\")
    while lines and lines[-1] == "\\addlinespace":
        lines.pop()

    with open(os.path.join(args.outdir, "rows2.tex"), "w") as f:
        f.write("\n".join(lines))
    h = np.array(half)
    print("CI half-width: median %.1f, IQR %.1f-%.1f, max %.1f points"
          % (np.median(h), *np.percentile(h, [25, 75]), h.max()))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
