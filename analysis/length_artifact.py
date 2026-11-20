#!/usr/bin/env python3
"""
Length-artifact analysis for the cloze-scored CCE runs (Section 4.4, Table 3).

In cloze scoring, each option's text is scored as a free continuation of the
question by its summed log-likelihood. "None of the above" (4 tokens) then
competes against options of very different lengths. This script shows how
the NOTA selection rate under cloze scoring tracks option length, and how it
changes when the same log-likelihoods are length-normalised (per character).

Usage:
    python analysis/length_artifact.py --root <cloze runs> [--seed 0] [--out length_artifact.csv]

Expected layout (what scripts/run_cloze.sh writes):
    <root>/<org>/<model>/cce_0shot_seed<N>/<org__model>/samples_<task>_<timestamp>.jsonl
Task names may carry the "_cloze_cce" suffix of tasks/cce_cloze (the paper's
runs used the bare dataset names); for each task the newest samples file is taken.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

NOTA = "none of the above"
TASKS = ["arc_challenge", "commonsense_qa", "hellaswag", "piqa", "winogrande"]


def ll(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def analyse(path):
    raw, norm, opt_len, gr, dr, gr_raw, dr_raw = [], [], [], [], [], [], []
    for line in open(path):
        r = json.loads(line)
        args = r["arguments"]
        args = list(args.values()) if isinstance(args, dict) else args
        conts = [a["arg_1"] if isinstance(a, dict) else a[1] for a in args]
        lls = np.array([ll(x) for x in r["filtered_resps"]])
        nota = [i for i, c in enumerate(conts) if c.strip().lower() == NOTA]
        if not nota:
            continue
        n = nota[0]
        lens = np.array([max(len(c.strip()), 1) for c in conts], dtype=float)
        pick_raw = int(np.argmax(lls)) == n
        raw.append(pick_raw)
        norm.append(int(np.argmax(lls / lens)) == n)
        opt_len.append(np.mean([lens[i] for i in range(len(conts)) if i != n]))
        regime = r["doc"].get("cce_regime")
        if regime == "gold_removed":
            gr_raw.append(pick_raw)
        elif regime == "distractor_removed":
            dr_raw.append(pick_raw)
    if not raw:
        return None
    m = lambda v: 100 * float(np.mean(v)) if v else float("nan")
    return dict(n=len(raw), mean_option_chars=float(np.mean(opt_len)),
                nota_rate_raw=m(raw), nota_rate_length_norm=m(norm),
                nota_rate_raw_GR=m(gr_raw), nota_rate_raw_DR=m(dr_raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="length_artifact.csv")
    args = ap.parse_args()

    rows = []
    pattern = os.path.join(os.path.expanduser(args.root), "**", f"cce_0shot_seed{args.seed}", "*", "samples_*.jsonl")
    files = {}
    for f in glob.glob(pattern, recursive=True):
        m = re.search(r"/([^/]+/[^/]+)/cce_0shot_seed\d+/[^/]+/samples_(.+?)_\d{4}-\d{2}-\d{2}T", f)
        task = m.group(2).removesuffix("_cloze_cce") if m else None
        if task not in TASKS:
            continue
        key = (m.group(1), task)
        if key not in files or os.path.basename(f) > os.path.basename(files[key]):
            files[key] = f
    print(f"{'model':38s} {'task':15s} {'opt chars':>9s} {'NOTA raw':>9s} {'(GR / DR)':>14s} {'len-norm':>9s}")
    for (model, task), f in sorted(files.items()):
        res = analyse(f)
        if res is None:
            continue
        rows.append(dict(model=model, task=task, **res))
        print(f"{model:38s} {task:15s} {res['mean_option_chars']:9.1f} {res['nota_rate_raw']:9.1f} "
              f"{res['nota_rate_raw_GR']:6.1f} / {res['nota_rate_raw_DR']:5.1f} {res['nota_rate_length_norm']:9.1f}")
    if rows:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.out}")
    else:
        print("No cloze-scored CCE samples found under", args.root)


if __name__ == "__main__":
    main()
