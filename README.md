# Counterfactual Choice Evaluation (CCE)

Code for **"Do LLMs Know When the Answer Is Absent? Counterfactual Choice
Evaluation of Multiple-Choice Benchmarks."**

Standard MCQA assumes the correct answer is always present, so high scores
can reflect option *ranking* rather than *verification*. CCE replaces one
option with **"None of the Above" (NOTA)**: if the replaced option was the
gold answer, NOTA becomes correct; otherwise the label is unchanged — probing
whether a model can tell when no option is right, with no new annotation.
Built on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

## Metrics

Each item is one of two regimes, reported separately:

- **GR** (gold-removed): correct option replaced; NOTA is the answer.
- **DR** (distractor-removed): wrong option replaced; original answer remains.
- **FP**: fraction of DR items where the model wrongly picks NOTA.

High GR with high FP = the model picks NOTA *indiscriminately*, not genuine
answer-absence reasoning — a confound aggregate accuracy hides. Tasks emit
`acc`, `acc_gold_removed`, `acc_distractor_removed`, `nota_false_positive`.

Tasks live under `lm_eval/tasks/{arc,commonsense_qa,hellaswag,piqa,winogrande}/`.
`winogrande_orig` is a matched baseline under the same prompt, so Orig–CCE
reflects only the NOTA substitution.

## Install

```bash
git clone https://github.com/<user>/<repo>.git   # TODO: repo URL
cd <repo> && pip install -e .
```

## Usage

The NOTA substitution is controlled by `CCE_SEED` (read in each task's
`utils.py`), separately from `--seed`. Vary it across runs to get the ±
reported in the paper.

```bash
MODEL=Qwen/Qwen3-8B
for s in 0 1 2 3 4; do
  CCE_SEED=$s srun --gres=gpu:4 python3 -m lm_eval --model hf \
    --model_args pretrained=${MODEL},parallelize=True,attn_implementation=sdpa \
    --tasks arc_challenge,commonsense_qa,hellaswag,piqa,winogrande,winogrande_orig \
    --device cuda --batch_size 16 --num_fewshot 0 --seed $s \
    --output_path results.mcq/${MODEL}/seed$s --log_samples
done
```

Use `--num_fewshot 5` for 5-shot CoT
