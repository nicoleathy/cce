# Can LLMs Detect Missing Answers? A Counterfactual Evaluation

Code and results for **Counterfactual Choice Evaluation (CCE)**, submitted to PAKDD 2027 (anonymous review).

Multiple-choice benchmarks always contain the correct answer, so a high score does not show whether a model notices when no option is correct. CCE replaces one option at a time with "None of the above" (NOTA), following the WiCkeD substitution (Elhady et al., 2025), and tests every possible replacement:

- **Gold-removed (GR):** the correct option was removed, so NOTA is correct.
- **Distractor-removed (DR):** a wrong option was removed, so the correct option is still present and NOTA is wrong.

The main measure is **NOTA discrimination** `D = ν_GR − ν_DR`, the NOTA selection rate in GR minus that in DR. A large `D` means the model detects a missing answer; `D ≈ 0` with similar GR and DR rates means it picks NOTA regardless of whether the answer is there (a preference, not detection). Because NOTA is always the last option, `ν_DR` is read against the model's last-option rate `L` on the original items.

## Repository layout

```
tasks/
  cce_mcf/            lm-eval tasks, letter scoring (all results except Table 3)
    *_mcf_orig        original items, same prompt and scoring
    *_mcf_cce         one random removal per item (5-shot design, seeds 0-4)
    *_mcf_exh         exhaustive design: every item x every removed option (0-shot)
    utils.py          prompts, WiCkeD substitution, random tie-breaking, metrics
  cce_cloze/          lm-eval tasks, cloze scoring of full option text (Table 3 only)
scripts/
  run_cce.sh          main runs for one model (Tables 1 and 2)
  run_all_models.sh   run_cce.sh over every model in models.tsv
  models.tsv          Hugging Face model ids and the exact commits used
  run_robustness.sh   chat template, float32, NOTA wording, no-BOS checks
  run_cloze.sh        cloze-scored runs (Table 3)
  make_qwen_nothink_tok.py   Qwen3.5 tokenizer copy with thinking disabled (chat check)
  slurm_example.sbatch       example cluster wrapper
analysis/
  paired_analysis.py  all Table 1/2 statistics from per-item sample logs
                      (paired bootstrap CIs, McNemar tests, coherence exclusion)
  coherence_flags.py  WiCkeD coherence classifier over every item
  length_artifact.py  cloze-scoring length analysis (Table 3)
  compare_runs.py     robustness run vs main run (Section 4.5)
  summarize_cce.py    quick summary and consistency check of results_*.json
  build_table{1,2,3}.py   LaTeX rows for Tables 1-3
results/
  tables/             the CSVs every number in the paper comes from (see below)
  main/               aggregated lm-eval output for the 16 models (0- and 5-shot)
  robustness/         chat/, fp32/, wording/, no_bos/
  cloze/              aggregated output of the cloze-scored runs
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login        # Llama and Gemma checkpoints are gated
```

The paper's runs used Python 3.11, `lm-evaluation-harness` 0.4.13, `transformers` 5.20.1 and bfloat16 weights. The tasks are loaded with `--include_path`, so a standard `lm_eval` install is enough.

## Reproducing the tables from the released results

The per-item sample logs are large and are not included, but every number in the paper is in `results/tables/`, so the tables can be rebuilt without a GPU:

```bash
python analysis/build_table1.py     # Table 1: accuracy, Orig vs CCE, 0- and 5-shot
python analysis/build_table2.py     # Table 2: GR, DR, D and L (0-shot), plus CI half-widths
python analysis/build_table3.py     # Table 3: cloze scoring and option length
```

| File in `results/tables/` | Contents | Used for |
|---|---|---|
| `cce_tables_final.csv` | all metrics per model, dataset and shot setting, with 95% bootstrap CIs and McNemar p-values | Tables 1-2, Sections 4.1-4.3 |
| `cce_tables_final_filtered.csv` | the same with coherence-flagged items removed | Section 4.5 (coherence) |
| `coherence_flags.csv`, `coherence_flags.log` | WiCkeD classifier output per item (`flippable = 0` means flagged) | Section 4.5 |
| `cce_chat.csv`, `cce_fp32.csv`, `cce_wording.csv` | robustness runs | Section 4.5 |
| `compare_chat.txt`, `compare_fp32.txt`, `compare_wording.txt` | robustness run vs main run | Section 4.5 |
| `length_artifact.csv`, `length_artifact.txt` | cloze NOTA rates, raw and length-normalised | Table 3, Section 4.4 |
| `paired_analysis_final*.txt` | full console output of `paired_analysis.py`, including tie rates | Sections 4.2, 4.5 |

`cce_tables_final.csv` has one row per (model, dataset, shots, setting, metric). `setting` is `orig`, `cce_seeds` (mean and SD over the five substitution seeds) or `exhaustive` (item-level means with bootstrap CIs). Metrics: `acc`, `A_GR` (= NOTA rate in GR), `A_DR`, `NOTA_DR`, `NOTA_discrimination` (`D`), `other_DR`, `delta_vs_orig`, `last_letter_rate` (`L`), `tie_rate`, `mcnemar_p_max`.

## Running the evaluation

### Main runs (Tables 1 and 2)

```bash
# one model at the paper's checkpoint
bash scripts/run_cce.sh meta-llama/Llama-3.2-3B runs/main 13afe5124825b4f3751f836b40dafda64c1ed062

# all 16 models (large models: PARALLELIZE=True BATCH_SIZE=2)
bash scripts/run_all_models.sh runs/main
```

For each model this writes `orig_{0,5}shot`, `cce_{0,5}shot_seed{0..4}` and `exh_0shot`. Then:

```bash
python analysis/paired_analysis.py --root runs/main --out cce_tables_final.csv
python analysis/coherence_flags.py --out coherence_flags.csv
python analysis/paired_analysis.py --root runs/main --exclude coherence_flags.csv \
       --out cce_tables_final_filtered.csv
python analysis/build_table1.py --csv cce_tables_final.csv
python analysis/build_table2.py --csv cce_tables_final.csv
```

### Robustness checks (Sections 4.3 and 4.5)

0-shot Orig and exhaustive CCE on ARC and WinoGrande, changing one factor at a time:

```bash
python scripts/make_qwen_nothink_tok.py --out qwen35_4b_nothink_tok   # needed for "chat"
bash scripts/run_robustness.sh chat    runs/chat      # native chat templates (4 IT models)
bash scripts/run_robustness.sh fp32    runs/fp32      # float32 scoring (4 IT models)
bash scripts/run_robustness.sh wording runs/wording   # "None of these options is correct" (3 models)
bash scripts/run_robustness.sh nobos   runs/no_bos    # no BOS token (Llama-3.2-1B)

python analysis/paired_analysis.py --root runs/chat --out cce_chat.csv
python analysis/compare_runs.py --base results/tables/cce_tables_final.csv --new cce_chat.csv
```

### Cloze scoring (Table 3)

```bash
for m in meta-llama/Llama-3.2-3B meta-llama/Llama-3.2-3B-Instruct \
         mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3; do
  bash scripts/run_cloze.sh "$m" runs/cloze
done
python analysis/length_artifact.py --root runs/cloze --seed 0 --out length_artifact.csv
python analysis/build_table3.py --csv length_artifact.csv
```

## Evaluation details

- **Prompt.** One plain prompt for every model, no chat template: the question or context, options labelled `A.`, `B.`, ..., then `Answer:`. WinoGrande is posed as a two-option question. The exact prompt builders are in `tasks/cce_mcf/utils.py`.
- **Substitution.** Removing option `j` and appending NOTA keeps the option count `k`; NOTA is always last. GR: the new gold is `k-1`. DR: the gold index shifts down by one if an earlier option was removed.
- **Scoring.** The answer letter with the highest log-likelihood is the prediction. Letters are scored instead of option text because summed log-likelihoods of text depend on its length (Table 3).
- **Ties.** bfloat16 scores can tie exactly. Ties are broken uniformly at random with a seed fixed per item and substitution (`zlib.crc32` of item id, removed index and regime), both in the task and again in `paired_analysis.py`. Always taking the first option would disadvantage NOTA.
- **BOS.** A BOS token is prepended for every model that defines one (`add_bos_token=True`).
- **Seeds.** 0-shot exhaustive runs test every removal. Random-removal runs use substitution seeds 0-4 (`CCE_SEED`), drawn per item so the draw does not depend on batching. Demonstrations come from the training split, are sampled with lm-eval few-shot seed 1234 and substituted with `CCE_FEWSHOT_SEED=1234`, so they are identical across seeds and models (about `1/k` have NOTA as the answer).
- **Statistics.** 0-shot: 95% bootstrap CIs from 2,000 item resamples, keeping each item's variants together. 5-shot: seed means, with an exact McNemar test per seed (bold in Table 1 requires `p < 0.05` in every seed).
- **Coherence.** Removing the gold may leave another acceptable answer. `coherence_flags.py` applies the WiCkeD classifier (`ahmedselhady/bert-base-uncased-sba-clf`) to each question; flagged items are excluded with `--exclude`.

### Benchmarks

| Benchmark | Split | Items | Options |
|---|---|---|---|
| ARC-Challenge (`allenai/ai2_arc`) | test | 1,172 | 3-5 |
| CommonsenseQA (`tau/commonsense_qa`) | validation | 1,221 | 5 |
| HellaSwag (`Rowan/hellaswag`) | validation | 10,042 | 4 |
| PIQA (`baber/piqa`) | validation | 1,838 | 2 |
| WinoGrande (`allenai/winogrande`, `winogrande_xl`) | validation | 1,267 | 2 |

### Models

All 16 models are evaluated 0-shot (exhaustive) and 5-shot (five substitution seeds). Exact commits are in `scripts/models.tsv`.

| Paper name | Hugging Face id |
|---|---|
| Llama-3.2-1B / 3B / 3B-IT | `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, `meta-llama/Llama-3.2-3B-Instruct` |
| Mistral-7B / 7B-IT | `mistralai/Mistral-7B-v0.3`, `mistralai/Mistral-7B-Instruct-v0.3` |
| Qwen3.5-2B / 4B / 9B | `Qwen/Qwen3.5-{2B,4B,9B}-Base` |
| Qwen3.5-2B-IT / 4B-IT / 9B-IT / 27B-IT | `Qwen/Qwen3.5-{2B,4B,9B,27B}` |
| Gemma-4-12B / 12B-IT / 31B / 31B-IT | `google/gemma-4-12B`, `google/gemma-4-12B-it`, `google/gemma-4-31B`, `google/gemma-4-31B-it` |

## Notes on the released results

- `results/main`, `results/robustness` and `results/cloze` hold lm-eval's aggregated `results_*.json` and run manifests, with local paths and environment dumps removed. The paper's numbers come from the per-item logs through `paired_analysis.py`, which re-applies random tie-breaking, so a few aggregated values in the JSON files can differ slightly from the tables.
- The Gemma-4-12B-IT float32 and chat-template runs used a later checkpoint than the main runs; the paper does not interpret them.
- `results/robustness/no_bos/` holds the Llama-3.2-1B runs without a BOS token (Section 4.3: NOTA on about 40% of WinoGrande items without BOS, about 1% with it).
- In the paper's cloze runs the tasks were named after the datasets (`arc_challenge`, ...). In this repository they are `<dataset>_cloze_cce` so they do not clash with lm-eval's built-in tasks; `length_artifact.py` accepts both names.

## Acknowledgements

The NOTA substitution and the coherence classifier come from WiCkeD (Elhady, Agirre and Artetxe, ACL 2025). Evaluation uses EleutherAI's `lm-evaluation-harness`.

## License

Released for anonymous review. A license will be added with the de-anonymised version.
