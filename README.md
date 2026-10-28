# Counterfactual Choice Evaluation (CCE)

Code for **"Can LLMs Detect Missing Answers? A Counterfactual Evaluation."**

Standard multiple-choice benchmarks guarantee that one option is correct, so a model can score well by *ranking* options without *verifying* that any is right. CCE removes this guarantee: building on the WiCkeD substitution, it removes one option and appends **"None of the above" (NOTA)** as the last option.

- **Gold-removed (GR):** the correct option was removed, so NOTA is correct.
- **Distractor-removed (DR):** a wrong option was removed; the answer is still present and NOTA is wrong.

A model that detects a missing answer picks NOTA far more often in GR than DR; one that likes NOTA (or the last position) picks it equally in both. Aggregate accuracy cannot tell these apart, so CCE scores the two regimes separately. No new annotation is needed.

Tasks run on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) via `--include_path`; no fork needed.

## Repository layout

```
tasks/cce_mcf/            lm-eval task definitions
  utils.py                prompt building, NOTA substitution, metrics
  *_mcf_orig.yaml         original items, same prompt and scoring
  *_mcf_cce.yaml          one random substitution per item (seeded)
  *_mcf_exh.yaml          exhaustive: every item x every removed option
scripts/
  run_cce.sh              runs all settings for one model
  paired_analysis.py      builds all tables from the per-item logs
  coherence_flags.py      flags possibly incoherent gold-removed items
requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login     # needed for gated models (e.g. Llama)
```

`requirements.txt` pins the paper's versions. Small-model results are sensitive to software details (see *Implementation notes*), so use these versions to reproduce the reported numbers.

## Benchmarks

| Task prefix | Hugging Face dataset | Split | Items | Options |
|---|---|---|---|---|
| `arc_challenge` | `allenai/ai2_arc` (ARC-Challenge) | test | 1,172 | 4 (a few 3 or 5) |
| `commonsense_qa` | `tau/commonsense_qa` | validation | 1,221 | 5 |
| `hellaswag` | `Rowan/hellaswag` | validation | 10,042 | 4 |
| `piqa` | `baber/piqa` | validation | 1,838 | 2 |
| `winogrande` | `allenai/winogrande` (winogrande_xl) | validation | 1,267 | 2 |

Datasets download automatically. Few-shot demonstrations come from each benchmark's training split.

## Prompt format

Options are listed with letters; the model is scored on the log-likelihood of the answer letter. Example gold-removed ARC item:

```
Question: Which of these is a renewable energy source?
A. Coal
B. Natural gas
C. Oil
D. None of the above
Answer:
```

The gold option ("Solar power") was removed, so the answer is **D**. The option count never changes. Each task family (`orig`, `cce`, `exh`) uses the same prompt and scoring, so Orig vs CCE differences reflect only the substitution.

## Quick start

Smoke test, 20 items per task (a few minutes on one GPU):

```bash
LIMIT=20 SEEDS=0 SHOTS_LIST=0 bash scripts/run_cce.sh meta-llama/Llama-3.2-1B results
```

Full run, all five benchmarks:

```bash
bash scripts/run_cce.sh Qwen/Qwen3.5-4B results
```

This produces `orig_{0,5}shot`, `cce_{0,5}shot_seed{0..4}` (one random substitution per item, five seeds), and `exh_0shot` (exhaustive; the paper's 0-shot results). Settings are environment variables (see the top of `scripts/run_cce.sh`):

```bash
BATCH_SIZE=4 bash scripts/run_cce.sh Qwen/Qwen3.5-9B results                     # less GPU memory
PARALLELIZE=True BATCH_SIZE=2 bash scripts/run_cce.sh Qwen/Qwen3.5-27B results   # split across GPUs
SHOTS_LIST=0 bash scripts/run_cce.sh google/gemma-4-31B-it results               # 0-shot only
LAUNCH="srun --gres=gpu:4" bash scripts/run_cce.sh <model> results               # under Slurm
```

On 24 GB GPUs, models up to ~9B fit on one card at `BATCH_SIZE=4`; larger models need `PARALLELIZE=True`, and very large vocabularies (Qwen3.5, Gemma 4) need smaller batches.

Results go to `results/<org>/<model>/<run>/`, with per-item logs (`samples_*.jsonl`) and a `run_manifest.txt` recording software versions and which runs finished or failed.

## Metrics

Each `cce`/`exh` task reports `acc` (all variants), `acc_gold_removed` (GR accuracy, i.e. the NOTA selection rate in GR), `acc_distractor_removed` (DR accuracy), `nota_rate_distractor` (NOTA chosen although the answer is present), `other_distractor_rate` (another wrong original option in DR), and `gold_removed_share`. Each `orig` task reports `acc` and `last_letter_rate`, the rate of choosing the last option when it is an ordinary candidate; since NOTA is always last, this separates a NOTA preference from a position preference.

The paper's main diagnostic is **NOTA discrimination**:

```
NOTA discrimination = P(NOTA | GR) - P(NOTA | DR)
```

A fixed NOTA preference gives a value near 0; recognising answer absence gives a value well above 0.

## Analysis

`paired_analysis.py` produces every number in the paper's tables from the per-item logs: Orig/CCE accuracy, regime metrics, NOTA discrimination, last-letter and tie rates, across-seed SDs, exact McNemar tests (per seed), and 95% item-level bootstrap intervals for the exhaustive design.

```bash
python scripts/paired_analysis.py --root results --out cce_tables.csv | tee paired_analysis.txt
```

### Coherence screen

Removing the gold option can leave a distractor that is also defensible. `coherence_flags.py` applies the WiCkeD coherence classifier to every item so the analysis can be repeated without them:

```bash
python scripts/coherence_flags.py --out coherence_flags.csv --device cpu
python scripts/paired_analysis.py --root results --exclude coherence_flags.csv --out cce_tables_filtered.csv
```

The classifier reads only the question text, so treat it as an approximate filter.

## Implementation notes

These choices affect results and are worth keeping if you extend the code.

- **Letter scoring.** Options are scored by the answer letter, not their text. Scoring option text as a free continuation makes the short NOTA string far more likely than long options, producing an artificial NOTA bias that depends on option length.
- **BOS token.** Always prepended (`add_bos_token=True`); small models' preferences change substantially without it.
- **Ties.** Scores are computed in bfloat16, so exact ties occur. They are broken uniformly at random with a fixed per-item seed; taking the first tied option would disadvantage NOTA, which is always last.
- **Seeds.** `CCE_SEED` sets the removed option in `cce` tasks (separate from lm-eval's `--seed`); each item's draw depends only on the seed and the item.
- **Few-shot demonstrations** are fixed training items (`CCE_FEWSHOT_SEED`) carrying the same substitution, so ~1/k of them have NOTA as the answer. Demonstrations end in a direct answer letter; no chain-of-thought.
- **No chat template** is applied, so all models see the same plain prompt; template-dependent models may be disadvantaged at 0-shot.
- **Exhaustive tasks** expand only the evaluation split; the few-shot pool is never expanded.

CCE builds on the WiCkeD substitution (Elhady, Agirre and Artetxe, ACL 2025); please cite their work as well.
