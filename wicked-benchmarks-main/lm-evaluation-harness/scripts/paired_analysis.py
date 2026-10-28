#!/usr/bin/env python3
"""
Build every CCE table from the per-item sample logs, so all numbers in the
paper come from one source.

Usage:
    python paired_analysis.py --root RESULTS_ROOT [--exclude coherence_flags.csv]
                              [--out cce_tables.csv] [--boot 2000]

Expected layout (what run_cce_mcf.sh writes):
    RESULTS_ROOT/<org>/<model>/<run>/<org__model>/samples_<task>_<timestamp>.jsonl
    <run> is orig_{S}shot, cce_{S}shot_seed{N} or exh_0shot

Per (model, dataset, shots) it reports:
    Orig       acc, last_letter_rate (preference for the last position when
               the last option is an ordinary one)
    CCE seeds  mean and across-seed SD (ddof=1) of acc, A_GR, A_DR, NOTA rate in
               DR, other-distractor rate in DR, NOTA discrimination, GR count,
               plus an exact McNemar test of Orig vs CCE correctness per seed
    Exhaustive item-level means with 95% bootstrap CIs (resampling items) for
               the same metrics and for the paired difference CCE - Orig

NOTA discrimination = P(NOTA | GR) - P(NOTA | DR). A model that picks NOTA
from a fixed preference scores ~0 regardless of how often it picks NOTA; a
model that tracks answer absence scores > 0.

With --exclude, items flagged by coherence_flags.py (flippable == 0) are
dropped from every setting, Orig included, so comparisons stay paired.
"""
import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np

RUN_RE = re.compile(
    r"(?:^|/)(?P<run>orig_(?P<so>\d+)shot|cce_(?P<sc>\d+)shot_seed(?P<seed>\d+)|exh_(?P<se>\d+)shot)"
    r"/(?P<mdir>[^/]+)/samples_(?P<task>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}T[^/]+)\.jsonl$"
)
TASK_RE = re.compile(r"^(?P<ds>.+)_mcf_(?P<kind>orig|cce|exh)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _ll(x):
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


def load_samples(path):
    """One dict per evaluated row: item, regime, gold, pred, k."""
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d = r["doc"]
            lls = [_ll(x) for x in r["filtered_resps"]]
            k = len(d["cce_choices"])
            if len(lls) != k:
                raise ValueError(f"{path}: doc {r['doc_id']} has {len(lls)} scores for {k} options")
            rows.append({
                "item": int(d.get("cce_item_id", r["doc_id"])),
                "regime": d["cce_regime"],
                "gold": int(d["cce_gold"]),
                "pred": int(np.argmax(lls)),
                "k": k,
            })
    return rows


def discover(root):
    """{(model, dataset, shots, kind, seed): newest sample file}."""
    found = {}
    for path in glob.glob(os.path.join(root, "**", "samples_*.jsonl"), recursive=True):
        m = RUN_RE.search(path.replace(os.sep, "/"))
        if not m:
            continue
        t = TASK_RE.match(m["task"])
        if not t:
            continue
        kind = t["kind"]
        shots = int(m["so"] or m["sc"] or m["se"])
        seed = int(m["seed"]) if m["seed"] is not None else None
        model = m["mdir"].replace("__", "/")
        key = (model, t["ds"], shots, kind, seed)
        if key not in found or os.path.basename(path) > os.path.basename(found[key]):
            found[key] = path  # keep the newest re-run
    return found


def load_exclusions(path):
    if not path:
        return {}
    excl = defaultdict(set)
    with open(path) as f:
        for r in csv.DictReader(f):
            if int(r["flippable"]) == 0:
                excl[r["dataset"]].add(int(r["item_id"]))
    return excl


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (binomial test on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(m + 1)]
    top = max(logs)
    tail = math.exp(top) * sum(math.exp(v - top) for v in logs)
    return min(1.0, 2 * tail)


def regime_metrics(rows):
    """Metrics over a list of CCE rows (one substitution per row)."""
    gr = [r for r in rows if r["regime"] == "gold_removed"]
    dr = [r for r in rows if r["regime"] == "distractor_removed"]
    nota = lambda r: r["pred"] == r["k"] - 1
    acc = np.mean([r["pred"] == r["gold"] for r in rows])
    a_gr = np.mean([r["pred"] == r["gold"] for r in gr]) if gr else float("nan")
    a_dr = np.mean([r["pred"] == r["gold"] for r in dr]) if dr else float("nan")
    nu_dr = np.mean([nota(r) for r in dr]) if dr else float("nan")
    other = np.mean([r["pred"] not in (r["gold"], r["k"] - 1) for r in dr]) if dr else float("nan")
    p = len(gr) / len(rows)
    implied = p * a_gr + (1 - p) * a_dr
    assert abs(implied - acc) < 1e-9, "mixture identity failed: rows are inconsistent"
    return {
        "acc": acc, "A_GR": a_gr, "A_DR": a_dr, "NOTA_DR": nu_dr,
        "other_DR": other, "NOTA_discrimination": a_gr - nu_dr,
        "n_GR": len(gr), "n_DR": len(dr),
    }


def bootstrap_ci(values, n_boot, rng, chunk=200):
    """95% percentile CI of the mean of per-item values (NaNs dropped)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, v.size, size=(b, v.size), dtype=np.int32)
        means.append(v[idx].mean(axis=1))
    means = np.concatenate(means)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--exclude", default=None, help="coherence_flags.csv from coherence_flags.py")
    ap.add_argument("--out", default="cce_tables.csv")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--rng_seed", type=int, default=0)
    args = ap.parse_args()

    files = discover(args.root)
    if not files:
        raise SystemExit(f"no samples_*_mcf_*.jsonl files found under {args.root}")
    excl = load_exclusions(args.exclude)
    rng = np.random.default_rng(args.rng_seed)

    groups = defaultdict(dict)  # (model, ds, shots) -> {(kind, seed): path}
    for (model, ds, shots, kind, seed), path in files.items():
        groups[(model, ds, shots)][(kind, seed)] = path

    out = []

    def emit(model, ds, shots, setting, metric, value, sd="", lo="", hi="", n=""):
        out.append(dict(model=model, dataset=ds, shots=shots, setting=setting, metric=metric,
                        value=value, sd_across_seeds=sd, ci95_lo=lo, ci95_hi=hi, n=n))

    for (model, ds, shots), runs in sorted(groups.items()):
        drop = excl.get(ds, set())
        keep = lambda rows: [r for r in rows if r["item"] not in drop]
        print(f"\n### {model} | {ds} | {shots}-shot" + (f" | {len(drop)} items excluded" if drop else ""))

        # Orig for the same shot setting (exhaustive runs are 0-shot, so they
        # pair with orig_0shot, which lives in the same group).
        orig = {}
        if ("orig", None) in runs:
            o = keep(load_samples(runs[("orig", None)]))
            orig = {r["item"]: r for r in o}
            acc = np.mean([r["pred"] == r["gold"] for r in o])
            last = np.mean([r["pred"] == r["k"] - 1 for r in o])
            print(f"  Orig: acc={100*acc:.1f}  last_letter_rate={100*last:.1f}  (n={len(o)})")
            emit(model, ds, shots, "orig", "acc", acc, n=len(o))
            emit(model, ds, shots, "orig", "last_letter_rate", last, n=len(o))

        # Random-substitution seeds
        seeds = sorted(s for (k, s) in runs if k == "cce")
        if seeds:
            per_seed = defaultdict(list)
            pvals = []
            for s in seeds:
                rows = keep(load_samples(runs[("cce", s)]))
                for name, v in regime_metrics(rows).items():
                    per_seed[name].append(v)
                if orig:
                    b = sum(1 for r in rows if r["item"] in orig
                            and orig[r["item"]]["pred"] == orig[r["item"]]["gold"] and r["pred"] != r["gold"])
                    c = sum(1 for r in rows if r["item"] in orig
                            and orig[r["item"]]["pred"] != orig[r["item"]]["gold"] and r["pred"] == r["gold"])
                    pvals.append(mcnemar_exact(b, c))
            parts = []
            for name, vals in per_seed.items():
                v = np.array(vals, dtype=float)
                sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
                emit(model, ds, shots, "cce_seeds", name, float(v.mean()), sd=sd, n=len(v))
                if name.startswith("n_"):
                    parts.append(f"{name}={[int(x) for x in v]}")
                else:
                    parts.append(f"{name}={100*v.mean():.1f}±{100*sd:.1f}")
            print(f"  CCE ({len(seeds)} seeds): " + ", ".join(parts))
            if pvals:
                print(f"  McNemar Orig vs CCE per seed: p = " + ", ".join(f"{p:.2g}" for p in pvals))
                emit(model, ds, shots, "cce_seeds", "mcnemar_p_max", max(pvals), n=len(pvals))

        # Exhaustive
        if ("exh", None) in runs:
            rows = keep(load_samples(runs[("exh", None)]))
            by_item = defaultdict(list)
            for r in rows:
                by_item[r["item"]].append(r)
            bad = [i for i, rs in by_item.items()
                   if len(rs) != rs[0]["k"] or sum(r["regime"] == "gold_removed" for r in rs) != 1]
            if bad:
                print(f"  WARNING: {len(bad)} items without exactly k variants and one GR variant")
            items = sorted(by_item)
            item_acc, item_gr, item_dr, item_nu, item_other, delta = [], [], [], [], [], []
            for i in items:
                rs = by_item[i]
                m = regime_metrics(rs)
                item_acc.append(m["acc"])
                item_gr.append(m["A_GR"])
                item_dr.append(m["A_DR"])
                item_nu.append(m["NOTA_DR"])
                item_other.append(m["other_DR"])
                if i in orig:
                    delta.append(m["acc"] - float(orig[i]["pred"] == orig[i]["gold"]))
            disc = np.array(item_gr) - np.array(item_nu)
            stats = {
                "acc": item_acc, "A_GR": item_gr, "A_DR": item_dr, "NOTA_DR": item_nu,
                "other_DR": item_other, "NOTA_discrimination": disc,
            }
            if delta:
                stats["delta_vs_orig"] = delta
            parts = []
            for name, vals in stats.items():
                mean, lo, hi = bootstrap_ci(vals, args.boot, rng)
                emit(model, ds, shots, "exhaustive", name, mean, lo=lo, hi=hi, n=len(items))
                parts.append(f"{name}={100*mean:.1f} [{100*lo:.1f}, {100*hi:.1f}]")
            print(f"  Exhaustive ({len(items)} items, {len(rows)} variants): " + ", ".join(parts))
            if not delta:
                print("  (no orig_0shot run found, so no paired difference)")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\nWrote {len(out)} rows to {args.out}")


if __name__ == "__main__":
    main()
