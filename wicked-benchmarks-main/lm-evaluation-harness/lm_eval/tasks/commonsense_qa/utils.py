# import copy
# import random
# from transformers import pipeline
# import torch
# from pprint import pprint

# random.seed(5331)
# ## classification pipeline:
# clf_pipeline = pipeline("text-classification", "ahmedselhady/bert-base-uncased-sba-clf", device="cpu") # "cuda" if torch.cuda.is_available() else "cpu")
# def can_be_flipped(question_text:str):
    
#     tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':512}

#     res = clf_pipeline(question_text, **tokenizer_kwargs)[0]
#     return res['label'].lower() == "negative"


# def preprocessing(dataset):
    
#     def process_doc(doc):
    
#         original_doc = copy.copy(doc)
#         if can_be_flipped(doc['question']):

#             # randomly select an answer to hide:
#             answer_to_hide = random.choice(doc['choices']['text'])      
#             answer_key = ord(original_doc['answerKey']) - 65
              
#             doc['choices']['text'].remove(answer_to_hide)
#             doc['choices']['text'].append("None of the above")
#             correct_answer = original_doc['choices']['text'][answer_key]
#             if correct_answer not in doc['choices']['text']:
#                 correct_answer = "None of the above"

#             doc['answerKey'] = chr(doc['choices']['text'].index(correct_answer)+65)
#         return doc
    
#     mapped_ds =  dataset.map(process_doc)
#     return mapped_ds

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


def preprocessing(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(_cce_seed())

    def process_doc(doc):
        texts = list(doc["choices"]["text"])
        labels = list(doc["choices"]["label"])

        gold_pos = labels.index(doc["answerKey"])
        gold_text = texts[gold_pos]

        replace_idx = rng.randrange(len(texts))
        texts[replace_idx] = NOTA

        if replace_idx == gold_pos:
            regime = "gold_removed"
            new_gold = replace_idx
        else:
            regime = "distractor_removed"
            new_gold = gold_pos          # gold slot untouched; no string lookup

        doc["choices"] = {"text": texts, "label": labels}
        doc["cce_choices"] = texts
        doc["gold"] = new_gold
        doc["cce_regime"] = regime
        doc["cce_replaced_index"] = replace_idx
        doc["cce_original_gold_index"] = gold_pos
        return doc

    return dataset.map(process_doc, load_from_cache_file=False)

