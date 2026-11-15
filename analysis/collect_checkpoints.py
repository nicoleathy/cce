#!/usr/bin/env python3
"""
List the exact model checkpoints (Hugging Face commit hashes) recorded by
lm-evaluation-harness in the aggregated results files.

Usage:
    python analysis/collect_checkpoints.py [--root results/lm_eval] [--out results/checkpoints.csv]

Warns if one model was evaluated with more than one revision.
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "lm_eval"))
    ap.add_argument("--out", default=os.path.join("results", "checkpoints.csv"))
    args = ap.parse_args()

    shas = defaultdict(set)
    params, setups = {}, defaultdict(set)
    for path in glob.glob(os.path.join(args.root, "**", "results_*.json"), recursive=True):
        cfg = json.load(open(path))["config"]
        margs = cfg["model_args"]
        model = margs["pretrained"] if isinstance(margs, dict) else margs.split("pretrained=")[1].split(",")[0]
        sha = cfg.get("model_sha") or "unknown"
        shas[model].add(sha)
        params[model] = cfg.get("model_num_parameters")
        parts = os.path.relpath(path, args.root).split(os.sep)
        setup = "/".join(parts[:2]) if parts[0] == "robustness" else parts[0]
        setups[(model, sha)].add(setup)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "revision", "num_parameters", "used_in"])
        for model in sorted(shas):
            if len(shas[model]) > 1:
                print(f"WARNING: {model} has {len(shas[model])} revisions: {sorted(shas[model])}")
            for sha in sorted(shas[model]):
                w.writerow([model, sha, params[model], " ".join(sorted(setups[(model, sha)]))])
    print(f"wrote {args.out} ({len(shas)} models)")


if __name__ == "__main__":
    main()
