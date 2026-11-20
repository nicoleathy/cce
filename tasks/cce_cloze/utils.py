"""
Cloze-scored CCE (Section 4.4, Table 3).

Each option is scored by the summed log-likelihood of its full text as a
continuation of the prompt. One option per item, drawn at random, is replaced
in place by "None of the above" (NOTA), so NOTA takes the removed option's
position:

    gold_removed       (replaced == gold): NOTA is correct
    distractor_removed (replaced != gold): the gold option is unchanged

The draw uses one RNG per split, seeded by the CCE_SEED environment variable
(Table 3 uses CCE_SEED=0). This variant exists only to show the length
artifact of cloze scoring; every other result in the paper uses the
letter-scored tasks in tasks/cce_mcf.
"""

import os
import random
import re

import datasets
import numpy as np

NOTA = "None of the above"


def _seed():
    return int(os.environ.get("CCE_SEED", "0"))


def nanmean(items):
    """Mean over non-NaN entries; 0.0 if a regime never occurs in the split."""
    arr = np.asarray(items, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def _substitute(options, gold, rng):
    """Replace one option in place by NOTA. Returns (options, new_gold, regime, j)."""
    options = list(options)
    j = rng.randrange(len(options))
    options[j] = NOTA
    regime = "gold_removed" if j == gold else "distractor_removed"
    return options, gold, regime, j  # NOTA sits at j, so the gold index is unchanged


def _cce_fields(options, gold, regime, j, orig_gold):
    return {
        "choices": options,
        "cce_choices": options,
        "gold": gold,
        "cce_regime": regime,
        "cce_replaced_index": j,
        "cce_original_gold_index": orig_gold,
    }


# ---------------------------------------------------------------------------
# Per-dataset process_docs
# ---------------------------------------------------------------------------

def _hs_clean(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")  # WikiHow artifacts
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _make(extract):
    def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
        rng = random.Random(_seed())

        def _proc(doc):
            query, options, gold = extract(doc)
            opts, new_gold, regime, j = _substitute(options, gold, rng)
            return {"query": query, **_cce_fields(opts, new_gold, regime, j, gold)}

        return dataset.map(_proc, load_from_cache_file=False)

    return process_docs


def _arc(doc):
    texts = list(doc["choices"]["text"])
    labels = list(doc["choices"]["label"])
    gold = labels.index(doc["answerKey"])
    return texts, labels, gold


def arc_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    # ARC lists the (substituted) options in the prompt, then scores option text.
    rng = random.Random(_seed())

    def _proc(doc):
        texts, labels, gold = _arc(doc)
        opts, new_gold, regime, j = _substitute(texts, gold, rng)
        query = f"Question: {doc['question']}\n"
        for text, label in zip(opts, labels):
            query += f"{label}. {text}\n"
        query += "Answer:"
        return {"query": query, **_cce_fields(opts, new_gold, regime, j, gold)}

    return dataset.map(_proc, load_from_cache_file=False)


csqa_process_docs = _make(lambda d: (
    f"Question: {d['question'].strip()}\nAnswer:",
    list(d["choices"]["text"]),
    list(d["choices"]["label"]).index(d["answerKey"]),
))

hellaswag_process_docs = _make(lambda d: (
    _hs_clean(d["activity_label"] + ": " + d["ctx_a"] + " " + d["ctx_b"].capitalize()),
    [_hs_clean(e) for e in d["endings"]],
    int(d["label"]),
))

piqa_process_docs = _make(lambda d: (
    f"Question: {d['goal']}\nAnswer:",
    [d["sol1"], d["sol2"]],
    int(d["label"]),
))

winogrande_process_docs = _make(lambda d: (
    f"Fill in the blank: {d['sentence']}\nWhich option best fills the blank \"_\"?\nAnswer:",
    [d["option1"], d["option2"]],
    int(d["answer"]) - 1,
))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def process_results(doc, results):
    """acc overall and per regime, plus NOTA chosen when the gold is present.
    length_artifact.py recomputes all Table 3 numbers from the sample logs."""
    lls = np.asarray([r[0] for r in results], dtype=float)
    pred = int(np.argmax(lls))
    gold = int(doc["gold"])
    acc = float(pred == gold)
    gr = doc["cce_regime"] == "gold_removed"
    nota = int(doc["cce_replaced_index"])
    nan = float("nan")
    return {
        "acc": acc,
        "acc_gold_removed": acc if gr else nan,
        "acc_distractor_removed": nan if gr else acc,
        "nota_false_positive": nan if gr else float(pred == nota),
    }
