#!/usr/bin/env bash
set -euo pipefail

TOKENIZER=llama2-7b-hf
TRAIN_FILE=fineweb-edu-100M
SCRIPT=./preprocess_dataset/tokenize_dataset.py
OUT_DIR=./tokenized_datasets/fineweb_llama2

BLOCK_SIZE="${BLOCK_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VALIDATION_SPLIT_PERCENTAGE="${VALIDATION_SPLIT_PERCENTAGE:-1}"
CACHE_DIR="${CACHE_DIR:-/root/autodl-tmp/cache}"

mkdir -p "$OUT_DIR"

echo "======================================"
echo "Starting preprocessing"
echo "Dataset: $TRAIN_FILE"
echo "Tokenizer: $TOKENIZER"
echo "Block size: $BLOCK_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Output: $OUT_DIR"
echo "======================================"

python "$SCRIPT" \
  --dataset_name "$TRAIN_FILE" \
  --tokenizer_name "$TOKENIZER" \
  --validation_split_percentage "$VALIDATION_SPLIT_PERCENTAGE" \
  --block_size "$BLOCK_SIZE" \
  --preprocessing_num_workers "$NUM_WORKERS" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUT_DIR" \
  --trust_remote_code

echo "======================================"
echo "Preprocessing finished"
echo "Tokenized dataset saved to:"
echo "$OUT_DIR"
echo "======================================"