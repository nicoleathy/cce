#!/usr/bin/env python3
"""
Flag items whose NOTA substitution may be incoherent, using the classifier
from the original WiCkeD code (ahmedselhady/bert-base-uncased-sba-clf; see the
commented-out can_be_flipped() in lm_eval/tasks/arc/utils.py). WiCkeD only
applied NOTA to items the classifier labels "negative" and left the rest
unchanged.

The classifier sees the question text only, not the options, and was trained
on WiCkeD's knowledge benchmarks. Treat its output as a first screen and back
it with a small human audit of gold-removed items.

Usage (from inside lm-evaluation-harness/):
    python ../scripts/coherence_flags.py --out ../coherence_flags.csv

Output CSV columns: dataset, item_id, flippable (1/0), label, score, text.
item_id matches cce_item_id in the sample logs, so paired_analysis.py can
drop flagged items with --exclude coherence_flags.csv.
"""
import argparse
import csv

import datasets
from transformers import pipeline

# Same dataset paths and evaluation splits as the cce_mcf yaml files.
SOURCES = {
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "commonsense_qa": ("tau/commonsense_qa", None, "validation"),
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "piqa": ("baber/piqa", None, "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
}


def question_text(name, doc):
    """The text WiCkeD's classifier would see: the question, without options."""
    if name in ("arc_challenge", "commonsense_qa"):
        return doc["question"]
    if name == "hellaswag":
        return doc["activity_label"] + ": " + doc["ctx_a"] + " " + doc["ctx_b"]
    if name == "piqa":
        return doc["goal"]
    return doc["sentence"]  # winogrande


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coherence_flags.csv")
    ap.add_argument("--datasets", default=",".join(SOURCES))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    clf = pipeline("text-classification", "ahmedselhady/bert-base-uncased-sba-clf",
                   device=args.device)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "item_id", "flippable", "label", "score", "text"])
        for name in args.datasets.split(","):
            path, config, split = SOURCES[name]
            ds = datasets.load_dataset(path, config, split=split)
            texts = [question_text(name, d) for d in ds]
            preds = clf(texts, batch_size=args.batch_size, truncation=True,
                        padding=True, max_length=512)
            n_flip = 0
            for i, (t, p) in enumerate(zip(texts, preds)):
                flip = int(p["label"].lower() == "negative")
                n_flip += flip
                w.writerow([name, i, flip, p["label"], f"{p['score']:.4f}", t[:200]])
            print(f"{name}: {len(texts)} items, {len(texts) - n_flip} flagged "
                  f"({100 * (len(texts) - n_flip) / len(texts):.1f}%)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
