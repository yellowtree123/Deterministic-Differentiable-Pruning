#!/bin/bash
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
python evaluation/eval.py \
  --model_path "/root/autodl-tmp/llama-7b-hf" \
  --eval_batchsize 32 \
  --zero_shot \
  --device cuda:0 \
  --tasks boolq piqa hellaswag winogrande arc_easy arc_challenge openbookqa\
  --log_dir "log_pruning" \
  --mask_path "/root/autodl-tmp/runs/llama-7b-hf_20260419_185441/2e-2_2e-2_4e-1/step_1000/trainable_params.pt" \

# mathqa boolq piqa hellaswag winogrande arc_easy arc_challenge openbookqa rte

# boolq piqa hellaswag winogrande arc_easy arc_challenge openbookqa rte
# mathqa
# pip install -U "datasets<4.0.0"