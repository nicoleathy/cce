# import os
# import random

# import datasets

# # Seed once at module load; override via env var to produce the five
# # WiCkeD/CCE variants the paper reports.
# random.seed(int(os.environ.get("CCE_SEED", "5331")))


# def _apply_cce(example):
#     """Replace one of the two options with 'None of the above' and adjust
#     the answer label.

#     WinoGrande answers are stored as the strings '1' or '2' (1-based).
#     After substitution there are still only two choices, one of which is
#     'None of the above'. The label points at whichever position now holds
#     the correct answer.
#     """
#     options = [example["option1"], example["option2"]]
#     correct_idx = int(example["answer"]) - 1  # 0-based gold index

#     replaced_idx = random.choice([0, 1])
#     options[replaced_idx] = "None of the above"

#     if replaced_idx == correct_idx:
#         # The gold option was the one replaced; NOTA now sits at that
#         # position and is the new correct answer.
#         new_idx = replaced_idx
#     else:
#         # A distractor was replaced; the original gold remains correct.
#         new_idx = correct_idx

#     return {
#         "sentence": example["sentence"],
#         "option1": options[0],
#         "option2": options[1],
#         "answer": str(new_idx + 1),  # back to 1-based
#     }


# def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
#     return dataset.map(_apply_cce)


# def doc_to_text(doc):
#     return f"Question: {doc['sentence']}\nAnswer:"


# def doc_to_target(doc):
#     idx = doc["sentence"].index("_")
#     options = [doc["option1"], doc["option2"]]
#     correct_idx = int(doc["answer"]) - 1
#     return doc["sentence"][:idx] + options[correct_idx]


# def doc_to_choice(doc):
#     idx = doc["sentence"].index("_")
#     options = [doc["option1"], doc["option2"]]
#     return [doc["sentence"][:idx] + opt for opt in options]


# __all__ = ["process_docs", "doc_to_text", "doc_to_target", "doc_to_choice"]


import random

import datasets

from cce_metrics import NOTA, cce_process_results, nanmean  # noqa: F401


SEED = 5331


def _fill(sentence: str, option: str) -> str:
    idx = sentence.index("_")
    return sentence[:idx] + option + sentence[idx + 1 :]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(SEED)

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

        # Build full-sentence continuations, one per (possibly NOTA) option.
        choices = [_fill(doc["sentence"], opt) for opt in options]

        return {
            "sentence": doc["sentence"],
            "choices": choices,
            "gold": new_gold,
            "cce_regime": regime,
            "cce_replaced_index": replace_idx,
            "cce_original_gold_index": gold_pos,
        }

    return dataset.map(_process_doc)


# doc_to_* helpers referenced by the YAML.
def doc_to_text(doc):
    return ""  # multiple_input style: full continuations live in `choices`


def doc_to_choice(doc):
    return doc["choices"]


def doc_to_target(doc):
    return doc["gold"]