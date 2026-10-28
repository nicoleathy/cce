"""
Counterfactual Choice Evaluation (CCE), multiple-choice-format (MCF) version.

Every item is shown with ALL options listed under letter labels, and the model
is scored on the log-likelihood of the answer letter (" A", " B", ...).

CCE substitution (WiCkeD, Elhady et al. 2025): remove option j and append
"None of the above" as the LAST option, so the option count k is unchanged and
"the above" refers to the options listed above it.

    gold_removed       (j == gold): NOTA (last option) is correct, y' = k - 1
    distractor_removed (j != gold): original gold kept; its index shifts down
                                    by one if the removed option came before it

Three task families share this file:

    *_mcf_orig  original items, identical prompt and scoring, no substitution
    *_mcf_cce   one random j per item, drawn from a per-item RNG seeded by
                CCE_SEED (the five-seed design)
    *_mcf_exh   exhaustive design: every item is expanded into k variants,
                one per possible j (one GR variant and k - 1 DR variants)

Few-shot demonstrations come from the train split. lm-eval runs process_docs on
that split too, so the train split is handled separately:

    * it is never expanded (exhaustive tasks fall back to one random j), and
    * its substitution is seeded by CCE_FEWSHOT_SEED (default 1234), which is
      fixed across runs, so changing CCE_SEED changes only the evaluated items
      and never the demonstrations.

Pair this with a fixed few-shot sampling seed in lm-eval
(--seed s,s,s,<fixed>) so the same demonstration items are drawn in every run.
"""

import os
import random
import re
import zlib

import datasets
import numpy as np

NOTA = "None of the above"
LETTERS = "ABCDEFGHIJ"

COLUMNS = (
    "cce_item_id",
    "cce_query",
    "cce_choices",
    "cce_letters",
    "cce_gold",
    "cce_regime",
    "cce_replaced_index",
    "cce_original_gold_index",
    "cce_num_options",
)


def _eval_seed():
    return int(os.environ.get("CCE_SEED", "0"))


def _fewshot_seed():
    return int(os.environ.get("CCE_FEWSHOT_SEED", "1234"))


def _is_fewshot_split(dataset):
    """True for the train split (the few-shot pool in every CCE yaml)."""
    split = getattr(dataset, "split", None)
    return split is not None and str(split).startswith("train")


def _item_rng(seed, item_id):
    """Per-item RNG: the draw for an item depends only on (seed, item_id),
    not on processing order, batching, or how many items came before it."""
    return random.Random(seed * 1_000_003 + item_id)


# ---------------------------------------------------------------------------
# Dataset-specific extraction: each returns (stem, options, gold_index)
# ---------------------------------------------------------------------------

def _arc(doc):
    texts = list(doc["choices"]["text"])
    labels = list(doc["choices"]["label"])
    return "Question: " + doc["question"].strip(), texts, labels.index(doc["answerKey"])


def _csqa(doc):
    texts = list(doc["choices"]["text"])
    labels = list(doc["choices"]["label"])
    return "Question: " + doc["question"].strip(), texts, labels.index(doc["answerKey"])


def _hs_clean(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")  # WikiHow artifacts
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def _hellaswag(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    stem = (
        "Context: " + _hs_clean(doc["activity_label"] + ": " + ctx)
        + "\nWhich ending is the most plausible continuation?"
    )
    return stem, [_hs_clean(e) for e in doc["endings"]], int(doc["label"])


def _piqa(doc):
    stem = "Goal: " + doc["goal"].strip() + "\nWhich solution achieves the goal?"
    return stem, [doc["sol1"], doc["sol2"]], int(doc["label"])


def _winogrande(doc):
    stem = (
        "Sentence: " + doc["sentence"].strip()
        + '\nWhich option best fills the blank "_"?'
    )
    return stem, [doc["option1"], doc["option2"]], int(doc["answer"]) - 1


EXTRACTORS = {
    "arc_challenge": _arc,
    "commonsense_qa": _csqa,
    "hellaswag": _hellaswag,
    "piqa": _piqa,
    "winogrande": _winogrande,
}


# ---------------------------------------------------------------------------
# Shared transformation
# ---------------------------------------------------------------------------

def _format(stem, options):
    lines = [stem] + [f"{LETTERS[i]}. {opt}" for i, opt in enumerate(options)]
    return "\n".join(lines) + "\nAnswer:"


def substitute(options, gold, j):
    """WiCkeD substitution for a given j. Returns (new_options, new_gold, regime)."""
    k = len(options)
    new_options = [o for i, o in enumerate(options) if i != j] + [NOTA]
    if j == gold:
        return new_options, k - 1, "gold_removed"
    return new_options, (gold if gold < j else gold - 1), "distractor_removed"


def _record(stem, options, gold, j, item_id):
    """One processed row. j=None means Orig (no substitution)."""
    if j is None:
        opts, new_gold, regime, j = list(options), gold, "orig", -1
    else:
        opts, new_gold, regime = substitute(options, gold, j)
    return {
        "cce_item_id": int(item_id),
        "cce_query": _format(stem, opts),
        "cce_choices": opts,
        "cce_letters": list(LETTERS[: len(opts)]),
        "cce_gold": int(new_gold),
        "cce_regime": regime,
        "cce_replaced_index": int(j),
        "cce_original_gold_index": int(gold),
        "cce_num_options": len(opts),
    }


def _make_single(extract, cce):
    """Orig (cce=False) or one random substitution per item (cce=True)."""

    def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
        seed = _fewshot_seed() if _is_fewshot_split(dataset) else _eval_seed()

        def _proc(doc, idx):
            stem, options, gold = extract(doc)
            j = _item_rng(seed, idx).randrange(len(options)) if cce else None
            return _record(stem, options, gold, j, idx)

        return dataset.map(_proc, with_indices=True, load_from_cache_file=False)

    return process_docs


def _make_exhaustive(extract):
    """Every item expanded into k variants (one per removed option)."""
    single = _make_single(extract, cce=True)

    def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
        if _is_fewshot_split(dataset):
            return single(dataset)  # few-shot pool: never expanded

        def _expand(batch, idxs):
            out = {c: [] for c in COLUMNS}
            for row, idx in enumerate(idxs):
                stem, options, gold = extract({k: v[row] for k, v in batch.items()})
                for j in range(len(options)):
                    rec = _record(stem, options, gold, j, idx)
                    for c in COLUMNS:
                        out[c].append(rec[c])
            return out

        return dataset.map(
            _expand,
            batched=True,
            with_indices=True,
            remove_columns=dataset.column_names,
            load_from_cache_file=False,
        )

    return process_docs


arc_orig = _make_single(_arc, cce=False)
arc_cce = _make_single(_arc, cce=True)
arc_exh = _make_exhaustive(_arc)
csqa_orig = _make_single(_csqa, cce=False)
csqa_cce = _make_single(_csqa, cce=True)
csqa_exh = _make_exhaustive(_csqa)
hellaswag_orig = _make_single(_hellaswag, cce=False)
hellaswag_cce = _make_single(_hellaswag, cce=True)
hellaswag_exh = _make_exhaustive(_hellaswag)
piqa_orig = _make_single(_piqa, cce=False)
piqa_cce = _make_single(_piqa, cce=True)
piqa_exh = _make_exhaustive(_piqa)
winogrande_orig = _make_single(_winogrande, cce=False)
winogrande_cce = _make_single(_winogrande, cce=True)
winogrande_exh = _make_exhaustive(_winogrande)


# ---------------------------------------------------------------------------
# Prompt hooks
# ---------------------------------------------------------------------------

def doc_to_text(doc):
    return doc["cce_query"]


def doc_to_choice(doc):
    return doc["cce_letters"]


def doc_to_target(doc):
    return int(doc["cce_gold"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nanmean(items):
    arr = np.asarray(items, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _pred(results, doc):
    """Highest-scoring option. Exact ties (common with bf16 scores) are broken
    at random with a seed fixed per item and substitution, so they neither
    favour earlier options nor disfavour NOTA, which is always last."""
    lls = [float(r[0]) for r in results]
    top = max(lls)
    tied = [i for i, v in enumerate(lls) if v == top]
    if len(tied) == 1:
        return tied[0]
    key = f"{doc['cce_item_id']}|{doc['cce_replaced_index']}|{doc['cce_regime']}"
    return random.Random(zlib.crc32(key.encode())).choice(tied)


def process_results_orig(doc, results):
    """acc, plus how often the LAST letter is picked when the last option is an
    ordinary one. Compare with the CCE NOTA rates, where NOTA is always last,
    to separate a NOTA preference from a last-position preference."""
    pred = _pred(results, doc)
    return {
        "acc": float(pred == int(doc["cce_gold"])),
        "last_letter_rate": float(pred == len(doc["cce_choices"]) - 1),
    }


def process_results_cce(doc, results):
    """Per-item metrics; NaN marks metrics that do not apply to this regime.

    acc_gold_removed        A_GR (equals the NOTA selection rate in GR)
    acc_distractor_removed  A_DR
    nota_rate_distractor    NOTA chosen although the gold is present
    other_distractor_rate   DR items where a remaining wrong option was chosen
    gold_removed_share      realised GR fraction
    """
    pred = _pred(results, doc)
    gold = int(doc["cce_gold"])
    nota = len(doc["cce_choices"]) - 1  # NOTA is always the last option
    gr = doc["cce_regime"] == "gold_removed"
    nan = float("nan")
    acc = float(pred == gold)
    return {
        "acc": acc,
        "acc_gold_removed": acc if gr else nan,
        "acc_distractor_removed": nan if gr else acc,
        "nota_rate_distractor": nan if gr else float(pred == nota),
        "other_distractor_rate": nan if gr else float(pred not in (gold, nota)),
        "gold_removed_share": float(gr),
    }
