# from datasets import load_dataset
# import random

# def apply_wicked_to_piqa(example):
#     # Original choices
#     sol1 = example["sol1"]
#     sol2 = example["sol2"]
#     correct_label = example["label"]

#     # Randomly select which option to replace with "None of the above"
#     replaced = random.choice(["sol1", "sol2"])

#     # Replace the chosen option
#     if replaced == "sol1":
#         new_sol1 = "None of the above"
#         new_sol2 = sol2
#         replaced_idx = 0
#     else:
#         new_sol1 = sol1
#         new_sol2 = "None of the above"
#         replaced_idx = 1

#     # Set new label
#     if replaced_idx == correct_label:
#         new_label = 2  # "None of the above" is correct
#     else:
#         new_label = correct_label

#     return {
#         "goal": example["goal"],
#         "sol1": new_sol1,
#         "sol2": new_sol2,
#         "label": new_label
#     }

# # Load PIQA dataset (use 'train' or 'validation')
# piqa = load_dataset("piqa", split="train")

# # Apply the WiCkeD transformation
# wicked_piqa = piqa.map(apply_wicked_to_piqa)

# # Display a few examples
# for i in range(3):
#     ex = wicked_piqa[i]
#     print(f"\nQ: {ex['goal']}")
#     print(f"A. {ex['sol1']}")
#     print(f"B. {ex['sol2']}")
#     correct = ["A", "B", "None of the above"][ex["label"]]
#     print(f"Correct Answer: {correct}")


import random

import datasets

from cce_metrics import NOTA, cce_process_results, nanmean  # noqa: F401


SEED = 5331


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(SEED)

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
            new_gold = choices.index(gold_text)

        return {
            "goal": doc["goal"],
            "choices": choices,
            "gold": new_gold,
            "cce_regime": regime,
            "cce_replaced_index": replace_idx,
            "cce_original_gold_index": gold_pos,
        }

    return dataset.map(_process_doc)
