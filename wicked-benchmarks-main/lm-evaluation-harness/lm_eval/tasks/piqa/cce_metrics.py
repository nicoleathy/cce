"""
Shared metric helpers for Counterfactual Choice Evaluation (CCE).

CCE replaces exactly one option of a multiple-choice item with the fixed
surface form "None of the above" (NOTA). Uniform random selection of the
replaced index induces two regimes, recorded per-item in doc["cce_regime"]:

    "gold_removed"      : the gold option was the one replaced, so NOTA is
                          now the correct answer and the model must detect
                          that no original option is valid.
    "distractor_removed": a wrong option was replaced; the gold answer is
                          retained and NOTA acts as an added distractor the
                          model must reject.

These helpers let a single evaluation pass report overall accuracy together
with accuracy stratified by regime, plus the NOTA false-positive rate
(distractor-removed items the model wrongly assigns to NOTA). They are wired
in through each task's process_results / metric_list; see the per-dataset
utils.py and YAML files.

Accuracy values are stored per item, using float("nan") for the regime that
does not apply to that item, and aggregated with nanmean so each item counts
only toward its own regime. acc_norm mirrors acc but uses the
length-normalised prediction.
"""

import numpy as np


NOTA = "None of the above"


def nanmean(items):
    """Mean over non-NaN entries; 0.0 if a regime never occurs in the split."""
    arr = np.asarray(items, dtype=float)
    if np.all(np.isnan(arr)):
        return 0.0
    return float(np.nanmean(arr))


def cce_process_results(doc, results):
    """Custom process_results for CCE multiple-choice tasks.

    Expects:
      - doc["gold"]        : int index of the correct choice in the (already
                             transformed) choice list.
      - doc["cce_regime"]  : "gold_removed" or "distractor_removed".
      - results            : list of (loglikelihood, is_greedy) pairs, one per
                             choice, in choice order (standard lm-eval
                             multiple_choice request layout).

    Returns overall acc / acc_norm plus regime-stratified copies and the
    NOTA false-positive indicator. Metrics not applicable to the item are
    emitted as NaN so nanmean ignores them.
    """
    lls, _is_greedy = zip(*results)
    lls = np.asarray(lls, dtype=float)

    # Length-normalised choices for acc_norm.
    choices = doc["choices"]
    completion_len = np.array([float(len(c)) for c in choices])

    gold = int(doc["gold"])
    pred = int(np.argmax(lls))
    pred_norm = int(np.argmax(lls / completion_len))

    acc = 1.0 if pred == gold else 0.0
    acc_norm = 1.0 if pred_norm == gold else 0.0

    regime = doc["cce_regime"]
    is_gold_removed = regime == "gold_removed"
    is_distractor_removed = regime == "distractor_removed"

    # Index of the NOTA option in this item's choice list (-1 if somehow absent).
    try:
        nota_idx = choices.index(NOTA)
    except ValueError:
        nota_idx = -1

    # NOTA false positive: only defined on distractor-removed items, where
    # picking NOTA is always wrong (the gold answer is still present).
    if is_distractor_removed:
        # On distractor-removed items the gold answer is still present, so
        # selecting NOTA is always a false positive.
        nota_fp = 1.0 if pred == nota_idx else 0.0
    else:
        nota_fp = float("nan")

    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "acc_gold_removed": acc if is_gold_removed else float("nan"),
        "acc_norm_gold_removed": acc_norm if is_gold_removed else float("nan"),
        "acc_distractor_removed": acc if is_distractor_removed else float("nan"),
        "acc_norm_distractor_removed": (
            acc_norm if is_distractor_removed else float("nan")
        ),
        "nota_false_positive": nota_fp,
    }
