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

    nota_idx = choices.index(NOTA) if NOTA in choices else -1

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
        options = [doc["option1"], doc["option2"]]
        gold_pos = int(doc["answer"]) - 1  # "1"/"2" -> 0/1
        gold_text = options[gold_pos]

        replace_idx = rng.randrange(len(options))
        options[replace_idx] = NOTA

        if replace_idx == gold_pos:
            regime = "gold_removed"
            new_gold = replace_idx
        else:
            regime = "distractor_removed"
            new_gold = options.index(gold_text)

        return {
            "sentence": doc["sentence"],
            "cce_choices": options,
            "gold": new_gold,
            "cce_regime": regime,
            "cce_replaced_index": replace_idx,
            "cce_original_gold_index": gold_pos,
        }

    return dataset.map(_process_doc, load_from_cache_file=False)


def doc_to_text(doc):
    return (
        f"Fill in the blank: {doc['sentence']}\n"
        f"Which option best fills the blank \"_\"?\nAnswer:"
    )


def doc_to_choice(doc):
    return doc["cce_choices"]


def doc_to_target(doc):
    return doc["gold"]


def process_docs_orig(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        options = [doc["option1"], doc["option2"]]
        gold_pos = int(doc["answer"]) - 1
        return {
            "sentence": doc["sentence"],
            "cce_choices": options,
            "gold": gold_pos,
        }

    return dataset.map(_process_doc, load_from_cache_file=False)


def orig_process_results(doc, results):
    lls, _is_greedy = zip(*results)
    lls = np.asarray(lls, dtype=float)
    gold = int(doc["gold"])
    pred = int(np.argmax(lls))
    return {"acc": 1.0 if pred == gold else 0.0}