#!/usr/bin/env python3
"""
Write the paper's Tables 1-3 as LaTeX from the result CSVs.

Usage:
    python analysis/make_tables.py [--results results] [--out results/tables]

Inputs:
    <results>/cce_tables.csv       from analysis/paired_analysis.py  (Tables 1, 2)
    <results>/length_artifact.csv  from analysis/length_artifact.py  (Table 3)

Numbers are percentages rounded half-up to one decimal. In Table 1, Diff is
computed from the rounded Orig and CCE values shown, so the columns are
consistent; significance comes from the unrounded paired statistics.
Table 1 needs \\usepackage{graphicx} (it is scaled to \\textwidth) and all
tables need \\usepackage{booktabs}.
"""
import argparse
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

# (name in the paper, Hugging Face id); None inserts a gap between families.
MODELS = [
    ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
    ("Llama-3.2-3B-IT", "meta-llama/Llama-3.2-3B-Instruct"),
    None,
    ("Mistral-7B", "mistralai/Mistral-7B-v0.3"),
    ("Mistral-7B-IT", "mistralai/Mistral-7B-Instruct-v0.3"),
    None,
    ("Qwen3.5-2B", "Qwen/Qwen3.5-2B-Base"),
    ("Qwen3.5-2B-IT", "Qwen/Qwen3.5-2B"),
    ("Qwen3.5-4B", "Qwen/Qwen3.5-4B-Base"),
    ("Qwen3.5-4B-IT", "Qwen/Qwen3.5-4B"),
    ("Qwen3.5-9B", "Qwen/Qwen3.5-9B-Base"),
    ("Qwen3.5-9B-IT", "Qwen/Qwen3.5-9B"),
    ("Qwen3.5-27B-IT", "Qwen/Qwen3.5-27B"),
    None,
    ("Gemma-4-12B", "google/gemma-4-12B"),
    ("Gemma-4-12B-IT$^{\\ddagger}$", "google/gemma-4-12B-it"),
    ("Gemma-4-31B", "google/gemma-4-31B"),
    ("Gemma-4-31B-IT", "google/gemma-4-31B-it"),
]
CLOZE_MODELS = [
    ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
    ("Llama-3.2-3B-IT", "meta-llama/Llama-3.2-3B-Instruct"),
    ("Mistral-7B", "mistralai/Mistral-7B-v0.3"),
    ("Mistral-7B-IT", "mistralai/Mistral-7B-Instruct-v0.3"),
]
DATASETS = [("ARC", "arc_challenge"), ("CSQA", "commonsense_qa"), ("HS", "hellaswag"),
            ("PIQA", "piqa"), ("WG", "winogrande")]
GROUP_HEAD = r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}"


def round1(x):
    """Round half-up to one decimal (x already in the target unit)."""
    return Decimal(repr(float(x))).quantize(Decimal("0.1"), ROUND_HALF_UP)


def fmt(pct, bold=False):
    """Format a percentage; LaTeX minus sign, optional bold."""
    s = str(round1(pct))
    if s == "-0.0":
        s = "0.0"
    if s.startswith("-"):
        s = ("{\\boldmath$-$}" if bold else "$-$") + s[1:]
    return f"\\textbf{{{s}}}" if bold else s


class Lookup:
    def __init__(self, path):
        d = pd.read_csv(path)
        self.d = d.set_index(["model", "dataset", "shots", "setting", "metric"]).sort_index()

    def __call__(self, model, ds, shots, setting, metric, col="value"):
        key = (model, ds, shots, setting, metric)
        if key not in self.d.index:
            return None
        return float(self.d.loc[key, col])


def table1(get):
    def block(shots):
        lines = []
        for entry in MODELS:
            if entry is None:
                lines.append("\\addlinespace")
                continue
            name, m = entry
            if get(m, "arc_challenge", shots, "orig", "acc") is None:
                continue  # 5-shot not run for the three largest models
            cells, diffs = [], []
            for _, ds in DATASETS:
                orig = get(m, ds, shots, "orig", "acc")
                if shots == 0:
                    cce = get(m, ds, 0, "exhaustive", "acc")
                    lo = get(m, ds, 0, "exhaustive", "delta_vs_orig", "ci95_lo")
                    hi = get(m, ds, 0, "exhaustive", "delta_vs_orig", "ci95_hi")
                    sig = lo > 0 or hi < 0
                else:
                    cce = get(m, ds, shots, "cce_seeds", "acc")
                    sig = get(m, ds, shots, "cce_seeds", "mcnemar_p_max") < 0.05
                diff = float(round1(100 * cce) - round1(100 * orig))
                diffs.append(diff)
                cells += [fmt(100 * orig), fmt(100 * cce), fmt(diff, sig)]
            cells.append(fmt(round1(sum(diffs) / len(diffs))))
            lines.append(f"{name:<29}& " + " & ".join(cells) + " \\\\")
        while lines and lines[-1] == "\\addlinespace":
            lines.pop()
        return lines

    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n, _ in DATASETS)
    return "\n".join([
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Accuracy (\%) on original items (Orig), under CCE, and their difference (Diff $=$ CCE $-$ Orig, computed from the values shown; positive indicates improvement). CCE replaces one option with ``None of the above'' and scores answer letters by log-likelihood, prepending BOS and breaking ties randomly with a fixed per-item seed. \emph{0-shot}: exhaustive removal, no substitution sampling. \emph{5-shot}: mean over five substitution seeds (max SD 1.9), using five training-set demonstrations, each with its own random substitution, identical across seeds and models, with direct answers; unavailable for Qwen3.5-27B-IT, Gemma-4-31B and Gemma-4-31B-IT. \textbf{Bold} Diff: significant at $p<0.05$ (0-shot: 95\% bootstrap CI of the paired difference excludes 0; 5-shot: exact McNemar test significant in every seed). $^{\ddagger}$: performs poorly with the plain prompt at 0-shot. IT: instruction-tuned; HS: HellaSwag; WG: WinoGrande.}",
        r"\label{tab:accuracy}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l rrr rrr rrr rrr rrr r@{}}",
        r"\toprule",
        "& " + head + r" & Avg \\",
        GROUP_HEAD,
        "Model" + " & Orig & CCE & Diff" * len(DATASETS) + r" & Diff \\",
        r"\midrule",
        r"Chance & $\approx$25.0 & & & 20.0 & & & 25.0 & & & 50.0 & & & 50.0 & & & \\",
        r"\midrule",
        r"\multicolumn{17}{@{}l}{\emph{0-shot}} \\",
        *block(0),
        r"\midrule",
        r"\multicolumn{17}{@{}l}{\emph{5-shot}} \\",
        *block(5),
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ])


def table2(get):
    lines, half_widths = [], []
    for entry in MODELS:
        if entry is None:
            lines.append("\\addlinespace")
            continue
        name, m = entry
        cells = []
        for _, ds in DATASETS:
            gr = get(m, ds, 0, "exhaustive", "A_GR")
            dr = get(m, ds, 0, "exhaustive", "NOTA_DR")
            lo = get(m, ds, 0, "exhaustive", "NOTA_discrimination", "ci95_lo")
            hi = get(m, ds, 0, "exhaustive", "NOTA_discrimination", "ci95_hi")
            last = get(m, ds, 0, "orig", "last_letter_rate")
            half_widths.append(100 * (hi - lo) / 2)
            cells += [fmt(100 * gr, bold=lo > 0), fmt(100 * dr), fmt(100 * last)]
        lines.append(f"{name:<29}& " + " & ".join(cells) + " \\\\")
    h = np.array(half_widths)
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for n, _ in DATASETS)
    tex = "\n".join([
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{NOTA selection rate (\%) in two 0-shot regimes, with the last-option rate $L$ (\%) on the original items. GR (gold removed): the correct answer is removed, so NOTA is correct and its selection rate equals accuracy. DR (distractor removed): a wrong option is removed, so NOTA is incorrect. Higher GR than DR indicates detection of missing answers; similar rates suggest a general NOTA preference. $L$ is a reference for position preference that also reflects option content and correctness. \textbf{Bold}: GR significantly exceeds DR (95\% bootstrap CI for the per-item difference excludes 0); median CI half-width for the gap: "
        + f"{np.median(h):.1f} points (maximum {h.max():.1f}; full intervals in the repository). Excluding coherence-flagged items changes no boldface. $^{{\\ddagger}}$: see Table~\\ref{{tab:accuracy}}. IT: instruction-tuned; HS: HellaSwag; WG: WinoGrande.}}",
        r"\label{tab:regime}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.9pt}",
        r"\begin{tabular}{@{}l rrr rrr rrr rrr rrr@{}}",
        r"\toprule",
        "& " + head + r" \\",
        GROUP_HEAD,
        "Model" + " & GR & DR & $L$" * len(DATASETS) + r" \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return tex


def table3(path, bold_at=85.0):
    d = pd.read_csv(path)
    chars = {ds: round(d[d.task == ds].mean_option_chars.iloc[0]) for _, ds in DATASETS}

    def cell(v):
        return fmt(v, bold=round1(v) >= Decimal(str(bold_at)))

    rows = []
    for name, m in CLOZE_MODELS:
        cells = []
        for _, ds in DATASETS:
            r = d[(d.model == m) & (d.task == ds)].iloc[0]
            cells += [cell(r.nota_rate_raw_GR), cell(r.nota_rate_raw_DR), cell(r.nota_rate_length_norm)]
        rows.append(f"{name:<16}& " + " & ".join(cells) + " \\\\")
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n} ({chars[ds]})}}" for n, ds in DATASETS)
    return "\n".join([
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{NOTA selection rate (\%) under cloze scoring, which scores each option by the log-likelihood of its full text (0-shot, one random removal per item, seed 0). Raw: summed log-likelihood, split into GR (correct answer removed) and DR (a wrong option removed). Norm: the same log-likelihoods divided by option length in characters, over all items. Parentheses: mean length of the remaining original options in characters (``None of the above'' has 17). \textbf{Bold}: NOTA chosen on at least "
        + f"{bold_at:g}" + r"\% of items. HS: HellaSwag; WG: WinoGrande.}",
        r"\label{tab:cloze}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\begin{tabular}{@{}l rrr rrr rrr rrr rrr@{}}",
        r"\toprule",
        "& " + head + r" \\",
        GROUP_HEAD,
        "&" + " & ".join([r" \multicolumn{2}{c}{Raw} & Norm"] * len(DATASETS)) + r" \\",
        "Model" + " & GR & DR &" * len(DATASETS) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=os.path.join("results", "tables"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    get = Lookup(os.path.join(args.results, "cce_tables.csv"))
    outputs = {
        "table1_accuracy.tex": table1(get),
        "table2_regimes.tex": table2(get),
        "table3_cloze.tex": table3(os.path.join(args.results, "length_artifact.csv")),
    }
    for name, tex in outputs.items():
        path = os.path.join(args.out, name)
        with open(path, "w") as f:
            f.write(tex)
        print("wrote", path)


if __name__ == "__main__":
    main()
