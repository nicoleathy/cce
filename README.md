# Can LLMs Detect Missing Answers? A Counterfactual Evaluation

Code and results for the paper above (under review).

**Counterfactual Choice Evaluation (CCE)** replaces one option of a multiple-choice
question with "None of the above" (NOTA) and tests every replacement:

- **GR (gold removed):** the correct answer is replaced, so NOTA is correct.
- **DR (distractor removed):** a wrong option is replaced, so the correct answer is still present.

We report NOTA discrimination `D = P(NOTA | GR) - P(NOTA | DR)`. A large `D` means the
model detects missing answers; `D ≈ 0` with high NOTA rates means a NOTA preference.
The substitution builds on WiCkeD (Elhady et al., 2025); evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) 0.4.13.

Benchmarks: ARC-Challenge, CommonsenseQA, HellaSwag, PIQA, WinoGrande.
Models: 16 open-weight Llama-3.2, Mistral-7B, Qwen3.5 and Gemma-4 models (1B to 31B).

## Layout

```
tasks/cce_mcf/     letter-scoring tasks: original, CCE (seeded), exhaustive
tasks/cce_cloze/   cloze-scoring tasks (Table 3)
scripts/           run_cce.sh, run_cloze.sh, run_robustness.sh, SLURM example
analysis/          paired_analysis.py, make_tables.py, compare_runs.py, ...
results/           CSVs behind every table, LaTeX tables, model revisions,
                   aggregated lm-eval results
```

| Paper | Source |
|---|---|
| Table 1 (accuracy, 0- and 5-shot) | `results/cce_tables.csv` |
| Table 2 (GR, DR, D, L) | `results/cce_tables.csv` |
| Table 3 (cloze scoring) | `results/length_artifact.csv` |
| Section 4.5 robustness | `results/robustness/`, `results/cce_tables_coherence_filtered.csv` |

## Setup

```bash
pip install -r requirements.txt   # lm_eval 0.4.13, transformers 5.20.1
hf auth login                     # Llama and Gemma are gated
```

## Reproduce

Rebuild Tables 1 to 3 from the shipped CSVs (no GPU):

```bash
python analysis/make_tables.py    # writes results/tables/*.tex
```

Rerun from scratch:

```bash
# Tables 1 and 2 (one model; repeat for each)
bash scripts/run_cce.sh meta-llama/Llama-3.2-3B outputs/main
python analysis/paired_analysis.py --root outputs/main --out results/cce_tables.csv
python analysis/paired_analysis.py --root outputs/main \
    --exclude results/coherence_flags.csv --out results/cce_tables_coherence_filtered.csv

# Table 3
bash scripts/run_cloze.sh meta-llama/Llama-3.2-3B outputs/cloze
python analysis/length_artifact.py --root outputs/cloze --out results/length_artifact.csv

# Section 4.5 (chat | fp32 | wording)
bash scripts/run_robustness.sh fp32
python analysis/paired_analysis.py --root outputs/robustness/fp32 --out fp32.csv
python analysis/compare_runs.py --base results/cce_tables.csv --new fp32.csv
```

Defaults match the paper (bfloat16, 0- and 5-shot, substitution seeds 0 to 4, fixed
demonstrations with seed 1234). Script headers list the options, including `REVISION`
to pin a checkpoint; exact revisions are in `results/checkpoints.csv`. The 27B and 31B
models were run with `PARALLELIZE=True BATCH_SIZE=2`.

## Notes

- Raw per-item logs (`samples_*.jsonl`, about 1.5 GB) are not included;
  rerunning the scripts regenerates them.
- Chat mode with Qwen3.5 needs a no-think tokenizer: `scripts/make_qwen_nothink_tok.py`.
- google/gemma-4-12B-it was run at revision `707f0a3b` (main runs) and `43c9d014`
  (robustness chat and float32 runs); the paper does not interpret the latter.
