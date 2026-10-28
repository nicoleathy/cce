"""Summarize CCE re-run results and check internal consistency.

Usage: python summarize_cce.py /path/to/results.mcf
Reads every results_*.json under the folder, prints per model/task/setting
the mean and across-seed SD of each metric, the exact GR count per seed,
and checks the mixture identity  acc = p*A_GR + (1-p)*A_DR  for every run.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

root = sys.argv[1]
runs = defaultdict(lambda: defaultdict(list))   # (model, task, shots) -> metric -> [values]
gr_counts = defaultdict(list)
bad = 0

for path in glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True):
    m = re.search(r"/(cce|orig)_(\d)shot(?:_seed(\d))?/", path)
    if not m:
        continue
    _, shots, _ = m.groups()
    data = json.load(open(path))
    model = data["config"]["model_args"].split("pretrained=")[1].split(",")[0]
    for task, res in data["results"].items():
        vals = {k.split(",")[0]: v for k, v in res.items()
                if isinstance(v, (int, float)) and "stderr" not in k}
        key = (model, task, shots)
        for k, v in vals.items():
            runs[key][k].append(v)
        if "gold_removed_share" in vals:
            n = data["n-samples"][task]["effective"]
            p = vals["gold_removed_share"]
            gr_counts[key].append(round(p * n))
            implied = p * vals["acc_gold_removed"] + (1 - p) * vals["acc_distractor_removed"]
            if abs(implied - vals["acc"]) > 1e-6:
                bad += 1
                print(f"IDENTITY FAILS: {key} acc={vals['acc']:.4f} implied={implied:.4f} ({path})")

for key in sorted(runs):
    model, task, shots = key
    parts = []
    for k, v in sorted(runs[key].items()):
        v = 100 * np.array(v)
        sd = f"±{v.std(ddof=1):.1f}" if len(v) > 1 else ""
        parts.append(f"{k}={v.mean():.1f}{sd}")
    extra = f" | GR counts {gr_counts[key]}" if key in gr_counts else ""
    print(f"{model} | {task} | {shots}-shot (n={len(next(iter(runs[key].values())))}) | " + ", ".join(parts) + extra)

print("\nAll runs satisfy the mixture identity." if bad == 0 else f"\n{bad} run(s) FAIL the identity check.")
