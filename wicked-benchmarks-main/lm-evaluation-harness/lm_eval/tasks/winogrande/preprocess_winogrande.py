def doc_to_text(doc):
    answer_to_num = {"1": 0, "2": 1}
    return answer_to_num[doc["answer"]]


def doc_to_target(doc):
    idx = doc["sentence"].index("_") + 1
    return doc["sentence"][idx:].strip()


def doc_to_choice(doc):
    idx = doc["sentence"].index("_")
    options = [doc["option1"], doc["option2"]]
    return [doc["sentence"][:idx] + opt for opt in options]

# from datasets import load_dataset, Dataset
# import pandas as pd
# import random

# def apply_wicked_to_winogrande(split="validation", seed=42):
#     random.seed(seed)
#     dataset = load_dataset("winogrande", "winogrande_xl", split=split)

#     wicked_data = {
#         "sentence": [],
#         "option1": [],
#         "option2": [],
#         "answer": [],
#         "__replaced_index": [],
#         "original_answer": [],
#     }

#     for ex in dataset:
#         sentence = ex["sentence"]
#         options = [ex["option1"], ex["option2"]]
#         correct_idx = int(ex["answer"]) - 1

#         replaced_index = random.choice([0, 1])
#         new_options = options.copy()
#         new_options[replaced_index] = "None of the above"

#         if replaced_index == correct_idx:
#             new_answer = replaced_index
#         else:
#             new_answer = correct_idx

#         wicked_data["sentence"].append(sentence)
#         wicked_data["option1"].append(new_options[0])
#         wicked_data["option2"].append(new_options[1])
#         wicked_data["answer"].append(str(new_answer + 1))  # 1-based
#         wicked_data["__replaced_index"].append(replaced_index)
#         wicked_data["original_answer"].append(ex["answer"])

#     df = pd.DataFrame(wicked_data)
#     return Dataset.from_pandas(df)

# wicked = apply_wicked_to_winogrande(split="validation")
# wicked.save_to_disk("winogrande_wicked_validation")

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



