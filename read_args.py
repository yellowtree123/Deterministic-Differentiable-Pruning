#!/usr/bin/env python3
import sys
import yaml
import shlex

DEFAULTS = {
    "model_name_or_path": "llama2-7b-hf",
    "tokenizer_name": "llama2-7b-hf",
    "dataset_name": "/root/autodl-tmp/tokenized_datasets/fineweb_llama2",
    "num_layers": 32,
    "num_experts": 64,
    "intermediate_size": 11008,
    "expert_intermediate_size": 1408,
    "num_attention_head": 32,
    "prunable_module": "head,intermediate",
}

if len(sys.argv) != 2:
    print("Usage: python read_args.py <config.yaml>", file=sys.stderr)
    sys.exit(1)

config_path = sys.argv[1]

with open(config_path, "r", encoding="utf-8") as f:
    user_cfg = yaml.safe_load(f) or {}

cfg = {**DEFAULTS, **user_cfg}

mapping = {
    "MODEL_A": cfg["model_name_or_path"],
    "TOKENIZER_B": cfg["tokenizer_name"],
    "TRAIN_FILE": cfg["dataset_name"],
    "NUM_LAYERS": cfg["num_layers"],
    "NUM_EXPERTS": cfg["num_experts"],
    "INTERMEDIATE_SIZE": cfg["intermediate_size"],
    "EXPERT_INTERMEDIATE_SIZE": cfg["expert_intermediate_size"],
    "NUM_ATTENTION_HEAD": cfg["num_attention_head"],
    "PRUNABLE_MODULE": cfg["prunable_module"],
}

for k, v in mapping.items():
    print(f'{k}={shlex.quote(str(v))}')