"""Shared constants for the table builders."""
from decimal import Decimal, ROUND_HALF_UP

# (name in the paper, Hugging Face id); None marks a group break in the table.
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

DATASETS = ["arc_challenge", "commonsense_qa", "hellaswag", "piqa", "winogrande"]

DEFAULT_TABLES = "results/tables/cce_tables_final.csv"


def round1(x):
    """Round half-up to one decimal place (Python's round() is half-to-even)."""
    return Decimal(repr(float(x))).quantize(Decimal("0.1"), ROUND_HALF_UP)


def pct1(x):
    """Fraction -> percentage string with one decimal, half-up."""
    s = str(round1(100 * float(x)))
    return "0.0" if s == "-0.0" else s
