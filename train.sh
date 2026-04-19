#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONFIG_FILE="${CONFIG_FILE:-./configs/llama-7b-hf.yaml}"
SCRIPT="${SCRIPT:-main.py}"

eval "$(python read_args.py "$CONFIG_FILE")"

# normalize prunable_module in case yaml has spaces like "head, intermediate"
PRUNABLE_MODULE="$(printf '%s' "$PRUNABLE_MODULE" | sed 's/[[:space:]]*,[[:space:]]*/,/g')"

# ========== TRAINING HYPERPARAMS ==========
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
PER_DEVICE_EVAL_BS="${PER_DEVICE_EVAL_BS:-2}"
GRAD_ACC="${GRAD_ACC:-2}"

WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
BLOCK_SIZE="${BLOCK_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"

TARGET_SPARSITY="${TARGET_SPARSITY:-0.60}"

# ========== L0 / SPARSITY ARGS ==========
RATIO="${RATIO:-2.0}"
TAU="${TAU:-1.0}"
EVAL_STEPS="${EVAL_STEPS:-100}"

# ========== DISTRIBUTED / ACCELERATE ==========
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"

RUN_ROOT="${OUT_DIR:-./runs/$(basename "$MODEL_A")_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT"

run_one() {
  local lr_z="$1"
  local lr_l13="$2"
  local lr_l2="$3"
  local tag="$4"
  local port="$5"
  local warmup="$6"
  local max_train_steps="$7"
  local target_sparsity="${8:-$TARGET_SPARSITY}"

  local out_dir="${RUN_ROOT}/${tag}"
  mkdir -p "$out_dir"

  echo "=================================================="
  echo "tag=$tag port=$port"
  echo "LR_Z=$lr_z  LR_L13=$lr_l13  LR_L2=$lr_l2"
  echo "warmup=$warmup  max_train_steps=$max_train_steps  target_sparsity=$target_sparsity"
  echo "MODEL_A=$MODEL_A"
  echo "TOKENIZER_B=$TOKENIZER_B"
  echo "TRAIN_FILE=$TRAIN_FILE"
  echo "PRUNABLE_MODULE=$PRUNABLE_MODULE"
  echo "OUT_DIR=$out_dir"
  echo "=================================================="

  accelerate launch \
    --num_processes "$NPROC_PER_NODE" \
    --main_process_port "$port" \
    "$SCRIPT" \
    --model_name_or_path "$MODEL_A" \
    --dataset_name "$TRAIN_FILE" \
    --trust_remote_code \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BS" \
    --gradient_accumulation_steps "$GRAD_ACC" \
    --z_learning_rate "$lr_z" \
    --lambda_13_learning_rate "$lr_l13" \
    --lambda_2_learning_rate "$lr_l2" \
    --weight_decay "$WEIGHT_DECAY" \
    --lr_scheduler_type cosine \
    --num_warmup_steps "$warmup" \
    --block_size "$BLOCK_SIZE" \
    --preprocessing_num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --output_dir "$out_dir" \
    --max_train_steps "$max_train_steps" \
    --with_tracking \
    --report_to tensorboard \
    --checkpointing_steps 50 \
    --sparsity \
    --prunable_module "$PRUNABLE_MODULE" \
    --target_sparsity "$target_sparsity" \
    --num_layers "$NUM_LAYERS" \
    --num_experts "$NUM_EXPERTS" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --expert_intermediate_size "$EXPERT_INTERMEDIATE_SIZE" \
    --num_attentention_head "$NUM_ATTENTION_HEAD" \
    --ratio "$RATIO" \
    --tau "$TAU" \
    --eval_steps "$EVAL_STEPS" \
    --distill
    # --enable_gradient_checkpointing \
}

# ========== SWEEP ==========
run_one "2e-2" "2e-2" "4e-1" "2e-2_2e-2_4e-1" "$MASTER_PORT" "100" "1000" "0.20"

