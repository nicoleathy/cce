#!/usr/bin/env python3
"""Table 1: accuracy on the original items (Orig) and under CCE, 0- and 5-shot.

    python analysis/build_table1.py [--csv results/tables/cce_tables_final.csv] [--outdir .]

0-shot CCE is the exhaustive design; bold marks a 95% bootstrap CI of the
paired difference that excludes 0. 5-shot CCE is the mean over five
substitution seeds; bold marks an exact McNemar p < 0.05 in every seed.
Diff is computed from the displayed (rounded) values, as in the paper.
Writes rows0.tex, rows5.tex (LaTeX rows) and table1_values.csv.
"""
import argparse
import os

import pandas as pd

from common import DATASETS, DEFAULT_TABLES, MODELS, round1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_TABLES)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)

    def get(m, ds, shots, setting, metric, col="value"):
        r = d[(d.model == m) & (d.dataset == ds) & (d.shots == shots)
              & (d.setting == setting) & (d.metric == metric)]
        return None if len(r) != 1 else r.iloc[0][col]

    def fmt(pct, bold=False):
        s = str(pct)
        if s == "-0.0":
            s = "0.0"
        if s.startswith("-"):
            s = ("{\\boldmath$-$}" if bold else "$-$") + s[1:]
        return f"\\textbf{{{s}}}" if bold else s

    def rows(shots):
        lines, records = [], []
        for e in MODELS:
            if e is None:
                lines.append("\\addlinespace")
                continue
            name, m = e
            if get(m, DATASETS[0], shots, "orig", "acc") is None:
                continue
            cells, diffs = [], []
            for ds in DATASETS:
                o = get(m, ds, shots, "orig", "acc")
                if shots == 0:
                    c = get(m, ds, 0, "exhaustive", "acc")
                    lo = get(m, ds, 0, "exhaustive", "delta_vs_orig", "ci95_lo")
                    hi = get(m, ds, 0, "exhaustive", "delta_vs_orig", "ci95_hi")
                    sig = lo > 0 or hi < 0
                else:
                    c = get(m, ds, shots, "cce_seeds", "acc")
                    sig = get(m, ds, shots, "cce_seeds", "mcnemar_p_max") < 0.05
                ro, rc = round1(100 * o), round1(100 * c)
                diff = rc - ro
                diffs.append(diff)
                cells += [fmt(ro), fmt(rc), fmt(diff, sig)]
                records.append((name, shots, ds, o, c, float(diff) / 100, sig))
            cells.append(fmt(round1(sum(diffs) / len(diffs))))
            lines.append(f"{name:<29}& " + " & ".join(cells) + " \\\\")
        while lines and lines[-1] == "\\addlinespace":
            lines.pop()
        return lines, records

    l0, r0 = rows(0)
    l5, r5 = rows(5)
    with open(os.path.join(args.outdir, "rows0.tex"), "w") as f:
        f.write("\n".join(l0))
    with open(os.path.join(args.outdir, "rows5.tex"), "w") as f:
        f.write("\n".join(l5))
    pd.DataFrame(r0 + r5, columns=["model", "shots", "ds", "orig", "cce", "diff", "sig"]).to_csv(
        os.path.join(args.outdir, "table1_values.csv"), index=False)
    print("\n".join(l0))
    print("---")
    print("\n".join(l5))


if __name__ == "__main__":
    main()
