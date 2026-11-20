#!/usr/bin/env python3
"""
Compare a robustness run (chat template, float32, NOTA wording) with the main
results.

Usage:
    python analysis/compare_runs.py --base results/tables/cce_tables_final.csv --new cce_chat.csv

For every (model, dataset) present in both files at 0-shot (exhaustive), prints
GR rate, DR NOTA rate and the GR-DR gap in each, the change in the gap, and
whether the gap is significant in each (95% CI excludes 0). Ends with the
summary numbers quoted in Section 4.5.
"""
import argparse
import pandas as pd


def table(path):
    d = pd.read_csv(path)
    d = d[(d.shots == 0) & (d.setting == "exhaustive")]
    v = d.pivot_table(index=["model", "dataset"], columns="metric", values="value")
    lo = d[d.metric == "NOTA_discrimination"].set_index(["model", "dataset"]).ci95_lo
    v["sig"] = lo > 0
    return v[["A_GR", "NOTA_DR", "NOTA_discrimination", "sig"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    b, n = table(args.base), table(args.new)
    j = b.join(n, lsuffix="_base", rsuffix="_new", how="inner")
    if j.empty:
        raise SystemExit("no overlapping model/dataset pairs; check model names")
    for c in ["A_GR", "NOTA_DR", "NOTA_discrimination"]:
        j[f"d_{c}"] = 100 * (j[f"{c}_new"] - j[f"{c}_base"])
    pd.set_option("display.width", 200)
    show = j.copy()
    for c in show.columns:
        if c.startswith(("A_GR", "NOTA")):
            show[c] = (100 * show[c]).round(1)
    print(show.round(1).to_string())
    flips = (j.sig_base != j.sig_new).sum()
    print(f"\nmax |change in GR-DR gap|: {j.d_NOTA_discrimination.abs().max():.1f} points")
    print(f"max |change in GR rate|:   {j.d_A_GR.abs().max():.1f} points")
    print(f"max |change in DR rate|:   {j.d_NOTA_DR.abs().max():.1f} points")
    print(f"significance changes:      {flips} of {len(j)} cells")


if __name__ == "__main__":
    main()
