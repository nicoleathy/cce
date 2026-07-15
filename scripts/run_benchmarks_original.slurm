#!/bin/bash
#SBATCH --job-name=mcq-bench-org
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/mcq-bench-original1.out
#SBATCH --error=.slurm/mcq-bench-original1.err

source /gscratch5/users/asalem/environments/latest-lmeval/bin/activate 

export HF_DATASETS_CACHE="/gscratch5/users/asalem/cache/hf_datasets_cache"
export TRANSFORMERS_CACHE="/gscratch5/users/asalem/transformers_cache/"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

models=(
    # lmsys/vicuna-33b-v1.3
    # "lmsys/vicuna-13b-v1.5"
    # lmsys/vicuna-7b-v1.5
    # "facebook/opt-30b"
    # "facebook/opt-iml-30b"
    # facebook/opt-13b
    # "tiiuae/Falcon3-10B-Base"
    # "tiiuae/Falcon3-10B-Instruct"
    # "tiiuae/Falcon3-7B-Base"
    # "tiiuae/Falcon3-7B-Instruct"
    # "codellama/CodeLlama-7b-hf"
    # codellama/CodeLlama-7b-Instruct-hf
    # "codellama/CodeLlama-13b-hf"
    # "codellama/CodeLlama-13b-Instruct-hf"
    # "codellama/CodeLlama-34b-hf"
    # "codellama/CodeLlama-34b-Instruct-hf"
    # "google-t5/t5-11b"
    # "google-t5/t5-3b"
    # "google/flan-t5-xxl"
    # "nlp-waseda/tacomet-gpt2-xl-english"
    # "mismayil/comet-bart-ai2"
    # "EleutherAI/gpt-neox-20b"
    # "EleutherAI/gpt-j-6b"
    # "EleutherAI/gpt-neo-2.7B"
    # allenai/unifiedqa-v2-t5-11b-1363200
    # "allenai/unifiedqa-v2-t5-3b-1363200"
    # "Qwen/Qwen2.5-7B"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B"
    # "Qwen/Qwen2.5-14B-Instruct"
    # Qwen/Qwen2.5-32B
    # Qwen/Qwen2.5-32B-Instruct
    # "Qwen/Qwen3-32B"
    # "Qwen/Qwen3-14B"
    # "Qwen/Qwen3-8B"
    # "meta-llama/Llama-3.2-1B"
    # "meta-llama/Llama-3.2-1B-Instruct"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "mistralai/Mistral-7B-v0.3"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Small-24B-Base-2501"
    # "mistralai/Mistral-Small-24B-Instruct-2501"
    # "google/gemma-3-12b-pt"
    # "google/gemma-3-12b-it"
    # "google/gemma-3-27b-pt"
    # "google/gemma-3-27b-it"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    # "microsoft/phi-4"

tasks_selected=(
    # "mmlu"
    # "mmlu_redux"
    # "mmlu_pro"
    # "truthfulqa_mc1"
    "commonsense_qa"
)

results_path=/gscratch5/users/asalem/wicked/results.mcq

for model in "${models[@]}"; do        
    for group_name in "${tasks_selected[@]}"; do

        num_fewshot=5
        python3 -m lm_eval \
            --model hf \
            --model_args pretrained=$model,parallelize=True,attn_implementation=sdpa \
            --tasks $group_name \
            --device cuda \
            --output_path ${results_path}/${model}/${group_name}_original_mc \
            --num_fewshot ${num_fewshot} \
            --log_samples
        
    done
    rm -rf /gscratch5/users/asalem/transformers_cache/models-*
done

