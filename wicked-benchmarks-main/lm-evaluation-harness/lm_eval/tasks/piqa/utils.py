import os
import random

import datasets
import numpy as np

NOTA = "None of the above"


def nanmean(items):
    """Mean over non-NaN entries; 0.0 if a regime never occurs in the split."""
    arr = np.asarray(items, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def cce_process_results(doc, results):
    lls, _is_greedy = zip(*results)
    lls = np.asarray(lls, dtype=float)

    choices = doc.get("cce_choices")
    if choices is None or len(choices) != len(lls):
        choices = [NOTA] * len(lls)
    gold = int(doc["gold"])
    pred = int(np.argmax(lls))

    acc = 1.0 if pred == gold else 0.0

    regime = doc["cce_regime"]
    is_gold_removed = regime == "gold_removed"
    is_distractor_removed = regime == "distractor_removed"

    nota_idx = -1
    for _i, _c in enumerate(choices):
        if _c == NOTA:
            nota_idx = _i
            break
    if nota_idx == -1:
        for _i, _c in enumerate(choices):
            if NOTA in _c:
                nota_idx = _i
                break

    if is_distractor_removed:
        nota_fp = 1.0 if pred == nota_idx else 0.0
    else:
        nota_fp = float("nan")

    return {
        "acc": acc,
        "acc_gold_removed": acc if is_gold_removed else float("nan"),
        "acc_distractor_removed": acc if is_distractor_removed else float("nan"),
        "nota_false_positive": nota_fp,
    }


def _cce_seed():
    """Seed for the NOTA-substitution RNG, read from the environment so each
    --seed in a sweep maps to a distinct, log-visible transform. Set CCE_SEED
    alongside lm-eval's --seed in the launch script."""
    return int(os.environ.get("CCE_SEED", "0"))


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(_cce_seed())

    def _process_doc(doc):
        choices = [doc["sol1"], doc["sol2"]]
        gold_pos = int(doc["label"])
        gold_text = choices[gold_pos]

        replace_idx = rng.randrange(len(choices))
        choices[replace_idx] = NOTA

        if replace_idx == gold_pos:
            regime = "gold_removed"
            new_gold = replace_idx
        else:
            regime = "distractor_removed"
            new_gold = gold_pos          # gold slot untouched; no string lookup

        return {
            "goal": doc["goal"],
            "choices": choices,
            "cce_choices": choices,
            "gold": new_gold,
            "cce_regime": regime,
            "cce_replaced_index": replace_idx,
            "cce_original_gold_index": gold_pos,
        }

    return dataset.map(_process_doc, load_from_cache_file=False)