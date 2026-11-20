#!/usr/bin/env python3
"""Table 3: NOTA selection under cloze scoring, raw and length-normalised.

    python analysis/build_table3.py [--csv results/tables/length_artifact.csv] [--outdir .]

The input is the output of analysis/length_artifact.py. Writes table3.tex.
"""
import argparse
import os

import pandas as pd

from common import round1

MODELS = [("Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
          ("Llama-3.2-3B-IT", "meta-llama/Llama-3.2-3B-Instruct"),
          ("Mistral-7B", "mistralai/Mistral-7B-v0.3"),
          ("Mistral-7B-IT", "mistralai/Mistral-7B-Instruct-v0.3")]
DS = [("ARC", "arc_challenge"), ("CSQA", "commonsense_qa"), ("HS", "hellaswag"),
      ("PIQA", "piqa"), ("WG", "winogrande")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/tables/length_artifact.csv")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    f = lambda x: str(round1(x))

    chars = {ds: round(d[d.task == ds].mean_option_chars.iloc[0]) for _, ds in DS}
    head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{n} ({chars[ds]})}}" for n, ds in DS)
    rows = []
    for name, m in MODELS:
        cells = []
        for _, ds in DS:
            r = d[(d.model == m) & (d.task == ds)].iloc[0]
            cells += [f(r.nota_rate_raw_GR), f(r.nota_rate_raw_DR), f(r.nota_rate_length_norm)]
        rows.append(f"{name:<16}& " + " & ".join(cells) + " \\\\")

    tex = r"""\begin{table}[ht!]
\centering
\caption{NOTA selection rate (\%) under cloze scoring, which scores each option by the log-likelihood of its full text (0-shot, one random removal per item, seed 0). Raw: summed log-likelihood, split into GR (correct answer removed) and DR (a wrong option removed). Norm: the same log-likelihoods divided by option length in characters, over all items. Parentheses: mean length of the remaining original options in characters (``None of the above'' has 17). HS: HellaSwag; WG: WinoGrande.}
\label{tab:cloze}
\scriptsize
\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}l rrr rrr rrr rrr rrr@{}}
\toprule
& """ + head + r""" \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}
& \multicolumn{2}{c}{Raw} & Norm & \multicolumn{2}{c}{Raw} & Norm & \multicolumn{2}{c}{Raw} & Norm & \multicolumn{2}{c}{Raw} & Norm & \multicolumn{2}{c}{Raw} & Norm \\
Model & GR & DR & & GR & DR & & GR & DR & & GR & DR & & GR & DR & \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    with open(os.path.join(args.outdir, "table3.tex"), "w") as fh:
        fh.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
