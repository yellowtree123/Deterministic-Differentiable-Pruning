# Deterministic-Differentiable-Pruning

Structured pruning of large language models via learnable L0 masks with Lagrangian relaxation and knowledge distillation. All model weights are frozen -- only lightweight mask parameters are trained to decide which attention heads and MLP intermediate dimensions to remove. You can check out our paper at [Deterministic Differentiable Structured Pruning for Large Language Models](https://arxiv.org/abs/2603.08065).

Supported architectures: **LLaMA / LLaMA-2**, **Qwen3**, **DeepSeek-MoE**.

---

## Project Structure

```
.
├── main.py                        # Mask learning (training) script
├── read_args.py                   # YAML config reader
├── train.sh                       # Training launcher (distributed via accelerate)
├── pruning.sh                     # Structural pruning launcher
├── configs/
│   └── llama-7b-hf.yaml          # Example config for LLaMA-7B
├── preprocess_dataset/
│   ├── tokenize_dataset.py        # Tokenization + chunking pipeline
│   └── tokenize.sh                # Tokenization launcher
├── actual_prune/
│   └── prune_and_save.py          # Physically remove pruned weights from model
├── evaluation/
│   ├── eval.py                    # Perplexity + zero-shot evaluation
│   ├── eval.sh                    # Evaluation launcher
│   ├── wikitext-2-raw-v1/         # Local WikiText-2 for perplexity eval
│   └── utils/
│       ├── process_args.py        # Evaluation argument parsing
│       └── utils.py               # Seed + parameter counting utilities
├── transformers/                  # Custom HF transformers fork (git submodule)
└── lm-evaluation-harness/         # lm-eval fork (git submodule)
```

---

## Installation

### 1. Clone the repository (with submodules)

```bash
git clone --recurse-submodules https://github.com/<your-username>/Deterministic-Differentiable-Pruning.git
cd Deterministic-Differentiable-Pruning
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

```bash
pip install torch accelerate datasets sentencepiece protobuf termcolor
pip install lm-eval==0.4.10
```

### 3. Install the custom transformers fork

This project requires a modified `transformers` library that accepts `head_z` / `intermediate_z` / `expert_z` mask arguments in model forward passes. Install it from the included submodule:

```bash
pip install -e ./transformers
```

---

## Quick Start: Pruning LLaMA-7B-HF

The full pipeline has four steps:

1. **Tokenize** the training dataset
2. **Train** the pruning masks (L0 mask learning)
3. **Evaluate** the pruned model (perplexity + zero-shot benchmarks)
4. **Structurally prune** the model (physically remove weights and save)

### Step 0: Prepare model and dataset

Download LLaMA-7B weights and a training dataset. The example below uses a subset of [FineWeb-Edu](https://huggingface.co/datasets/codelion/fineweb-edu-100M):

---

### Step 1: Tokenize the dataset

Tokenize and chunk the raw dataset into fixed-length blocks before training. This avoids repeated tokenization during training.

Edit `preprocess_dataset/tokenize.sh` to set your paths, then run:

```bash
bash preprocess_dataset/tokenize.sh
```

This produces a HuggingFace `datasets` directory at `./tokenized_datasets/fineweb_llama` with `train` and `validation` splits, where each sample has `input_ids` and `labels` of length `block_size`.

**Key arguments:**

| Argument                        | Description                                                  |
| ------------------------------- | ------------------------------------------------------------ |
| `--dataset_name`                | Path to raw dataset (local or HuggingFace hub name)          |
| `--tokenizer_name`              | Path to tokenizer (must match the model you will prune)      |
| `--block_size`                  | Sequence length for chunking (default: model max, recommended: 2048) |
| `--validation_split_percentage` | Percentage of data to hold out for validation                |
| `--output_dir`                  | Where to save the tokenized dataset                          |

---

### Step 2: Train (learn pruning masks)

Create a config YAML for your model. Example `configs/llama-7b-hf.yaml`:

```yaml
model_name_or_path: ./llama-7b-hf
tokenizer_name: ./llama-7b-hf
dataset_name: ./tokenized_datasets/fineweb_llama

num_layers: 32
intermediate_size: 11008
num_attention_head: 32

prunable_module: head,intermediate
```

For MoE models like Deepseek-MoE-16B-Base change the yaml accordingly:

```yaml
num_layers: 27
num_experts: 64
expert_intermediate_size: 1408

prunable_module: expert
```

Launch training with:

```bash
bash train.sh ./configs/llama-7b-hf.yaml
```

The default configuration in `train.sh` runs a single sweep: 1000 steps, target sparsity 0.20, with distillation enabled. Checkpoints (containing only the learned mask parameters) are saved every 50 steps to `./runs/<model>_<timestamp>/<tag>/step_<N>/trainable_params.pt`.

**Key environment variables for `train.sh`:**

| Variable          | Default                      | Description                                                |
| ----------------- | ---------------------------- | ---------------------------------------------------------- |
| `CONFIG_FILE`     | `./configs/llama-7b-hf.yaml` | Path to model/data config                                  |
| `TARGET_SPARSITY` | `0.60`                       | Target fraction of parameters to prune (0.20 = remove 20%) |
| `NPROC_PER_NODE`  | `4`                          | Number of GPUs                                             |
| `PER_DEVICE_BS`   | `2`                          | Per-device training batch size                             |
| `GRAD_ACC`        | `2`                          | Gradient accumulation steps                                |
| `WARMUP_STEPS`    | `100`                        | LR warmup steps                                            |
| `EVAL_STEPS`      | `100`                        | Evaluate every N optimizer steps                           |
| `RATIO`           | `2.0`                        | Distillation loss coefficient                              |
| `TAU`             | `1.0`                        | Distillation temperature                                   |
| `SEED`            | `42`                         | Random seed                                                |

**What happens during training:**
- All model weights are **frozen** -- only mask parameters (`z_loga`) and Lagrangian dual variables are optimized.
- Three loss terms are minimized jointly: language modeling loss, Lagrangian sparsity penalty, and (optionally) knowledge distillation KL-divergence.
- Output: `trainable_params.pt` -- a lightweight checkpoint containing only the learned masks.

---

### Step 3: Evaluate the pruned model

Evaluate perplexity (WikiText-2) and zero-shot benchmarks by applying the learned masks as soft multipliers on model weights:

```bash
python evaluation/eval.py \
  --model_path ./llama-7b-hf \
  --mask_path ./runs/<your_run>/step_1000/trainable_params.pt \
  --device cuda:0 \
  --eval_batchsize 32 \
  --zero_shot \
  --tasks boolq piqa hellaswag winogrande arc_easy arc_challenge openbookqa \
  --log_dir ./log_pruning
```

Or edit and run the launcher:

```bash
bash evaluation/eval.sh
```

This first computes WikiText-2 perplexity, then runs the specified `lm-eval` zero-shot tasks. Results are printed to stdout and saved to `--log_dir`.

**Key arguments:**

| Argument           | Description                                 |
| ------------------ | ------------------------------------------- |
| `--model_path`     | Path to the original (unpruned) model       |
| `--mask_path`      | Path to `trainable_params.pt` from training |
| `--zero_shot`      | Run zero-shot benchmarks via lm-eval        |
| `--tasks`          | Space-separated list of lm-eval task names  |
| `--eval_batchsize` | Batch size for zero-shot evaluation         |
| `--device`         | Device to use (e.g. `cuda:0`)               |

---

### Step 4: Structural pruning (save the pruned model)

Once you are satisfied with the evaluation results, physically remove the pruned weights to produce a smaller model:

```bash
python actual_prune/prune_and_save.py \
  --state_dict_path ./runs/<your_run>/step_1000/trainable_params.pt \
  --model_name_or_path ./llama-7b-hf \
  --output_dir ./llama-7b-hf-pruned \
  --trust_remote_code \
  --safe_serialization
```

This script:
1. Loads the learned mask checkpoint
2. Applies `ReLU` to get binary indicators (nonzero = keep, zero = prune)
3. For each layer: physically removes pruned attention heads (Q/K/V/O projections) and MLP dimensions (gate/up/down projections)
4. Scales remaining weights by their mask values before pruning
5. Saves the structurally smaller model to `--output_dir`

The output is a standard HuggingFace model directory that can be loaded directly:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./llama-7b-hf-pruned")
```

**Key arguments:**

| Argument               | Description                                 |
| ---------------------- | ------------------------------------------- |
| `--state_dict_path`    | Path to `trainable_params.pt` from training |
| `--model_name_or_path` | Path to the original dense model            |
| `--output_dir`         | Where to save the pruned model              |
| `--safe_serialization` | Save in safetensors format                  |

---

## Configuration Reference

The YAML config file (e.g. `configs/llama-7b-hf.yaml`) defines model architecture and data paths:

| Field                      | Description                                                  | LLaMA-7B Value                       |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------ |
| `model_name_or_path`       | Path to pretrained model                                     | `./llama-7b-hf`                      |
| `tokenizer_name`           | Path to tokenizer                                            | `./llama-7b-hf`                      |
| `dataset_name`             | Path to pre-tokenized dataset                                | `./tokenized_datasets/fineweb_llama` |
| `num_layers`               | Number of transformer layers                                 | `32`                                 |
| `intermediate_size`        | MLP intermediate dimension                                   | `11008`                              |
| `num_attention_head`       | Number of attention heads                                    | `32`                                 |
| `num_experts`              | Number of MoE experts (set for MoE models)                   | `64`                                 |
| `expert_intermediate_size` | Expert MLP dimension (set for MoE models)                    | `1408`                               |
| `prunable_module`          | Comma-separated modules to prune: `head`, `intermediate`, `expert` | `head,intermediate`                  |
