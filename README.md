# Counterfactual Choice Evaluation (CCE)

Code and results for the paper **"Counterfactual Choice Evaluation"**.

CCE replaces one option of a multiple-choice item with "None of the above" (NOTA).
If the gold answer is removed (GR), NOTA becomes correct; if a distractor is removed (DR),
the gold answer remains. We report NOTA discrimination `D = P(NOTA | GR) - P(NOTA | DR)`.
The substitution builds on WiCkeD (Elhady et al.); evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Layout

```
tasks/cce_mcf/     letter-scoring tasks (orig, cce, exhaustive)
tasks/cce_cloze/   cloze-scoring tasks
scripts/           run_cce.sh, run_cloze.sh, run_robustness.sh, SLURM example
analysis/          paired_analysis.py, make_tables.py, compare_runs.py, ...
results/           CSVs behind every table, LaTeX tables, model revisions
```

## Setup

```bash
pip install -r requirements.txt   # lm_eval 0.4.13, transformers 5.20.1
hf auth login                     # Llama and Gemma are gated
```

## Reproduce

Rebuild Tables 1 to 3 from the shipped CSVs (no GPU):

```bash
python analysis/make_tables.py
```

Rerun from scratch:

```bash
# Tables 1 and 2
bash scripts/run_cce.sh meta-llama/Llama-3.2-3B outputs/main
python analysis/paired_analysis.py --root outputs/main --out results/cce_tables.csv

# Table 3
bash scripts/run_cloze.sh meta-llama/Llama-3.2-3B outputs/cloze
python analysis/length_artifact.py --root outputs/cloze --out results/length_artifact.csv

# Section 4.5 (chat | fp32 | wording)
bash scripts/run_robustness.sh fp32
python analysis/paired_analysis.py --root outputs/robustness/fp32 --out fp32.csv
python analysis/compare_runs.py --base results/cce_tables.csv --new fp32.csv
```

Script headers list the options (seeds, batch size, `REVISION`, `LIMIT`, ...).
Defaults match the paper. Exact model revisions are in `results/checkpoints.csv`.

## Notes

- Raw per-item logs (~1.4 GB) are not included; rerunning the scripts regenerates them.
- Chat mode with Qwen3.5 needs a no-think tokenizer: `scripts/make_qwen_nothink_tok.py`.
- google/gemma-4-12B-it was run at revision `707f0a3b` (main) and `43c9d014` (robustness chat and fp32).
