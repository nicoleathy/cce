#!/usr/bin/env python3
"""Save a copy of a Qwen3.5 tokenizer whose chat template always closes the
<think> block, so chat-template prompts end with '<think>\\n\\n</think>\\n\\n'
and the answer letter is scored directly after the assistant turn.

Usage:
    python scripts/make_qwen_nothink_tok.py --out qwen35_4b_nothink_tok
    QWEN_NOTHINK_TOK=qwen35_4b_nothink_tok bash scripts/run_robustness.sh chat
"""
import argparse

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.chat_template = "{%- set enable_thinking = false -%}\n" + tok.chat_template
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}],
                                     tokenize=False, add_generation_prompt=True)
    print("prompt ends with:", repr(prompt[-40:]))
    assert prompt.rstrip("\n").endswith("</think>"), "template did not close <think>"
    tok.save_pretrained(args.out)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
