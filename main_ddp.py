#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0

"""
Torchrun + native PyTorch DDP version of the original Accelerate script.

Launch (single node):
  torchrun --standalone --nproc_per_node=8 run_clm_no_trainer_torchrun.py ...

Notes:
- This removes Accelerate entirely.
- "resume_from_checkpoint" supports loading ONLY your saved trainable params
  (trainable_params.pt) because your script saves only those (not optimizer/scheduler).
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import datasets
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers

from datasets import load_dataset
from huggingface_hub import HfApi
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import contextmanager, nullcontext
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# -------------------------
# DDP utilities (torchrun)
# -------------------------

def ddp_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def ddp_init():
    """
    torchrun sets:
      RANK, WORLD_SIZE, LOCAL_RANK (and MASTER_ADDR, MASTER_PORT)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    return rank, world_size, local_rank, device


def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()


def ddp_all_reduce_(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    if ddp_is_initialized():
        dist.all_reduce(t, op=op)
    return t


def ddp_reduce_mean_scalar(t: torch.Tensor, world_size: int) -> torch.Tensor:
    # expects scalar tensor on correct device
    ddp_all_reduce_(t, op=dist.ReduceOp.SUM)
    return t / max(int(world_size), 1)


@contextmanager
def main_process_first(rank: int):
    """
    Mimic accelerate.main_process_first():
      - non-rank0 waits
      - rank0 executes
      - barrier
      - others proceed
    """
    if ddp_is_initialized() and rank != 0:
        dist.barrier()
    yield
    if ddp_is_initialized() and rank == 0:
        dist.barrier()


def print_rank0(is_main: bool, *args, **kwargs):
    if is_main:
        print(*args, **kwargs)


def set_seed(seed: int, rank: int = 0):
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# -------------------------
# Your L0 / Mask code
# -------------------------

@dataclass
class L0Config:
    pruning_modules: List[str] = field(default_factory=list)  # e.g. ["head", "intermediate"]
    target_sparsity: float = 0.20
    num_layers: int = 16
    num_experts: int = 64
    intermediate_size: int = 1024
    expert_intermediate_size: int = 1024
    num_attentention_head: int = 1024
    uniform_sparsity: bool = False

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)


limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6


def ste_clamp(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    x_hard = torch.clamp(x, min_val, max_val)
    return x + (x_hard - x).detach()


def ste_relu(x: torch.Tensor) -> torch.Tensor:
    x_hard = torch.relu(x)
    return x + (x_hard - x).detach()


class Mask(nn.Module):
    def __init__(
        self,
        name: str,
        mask_shape: List,
        mask_output_shape: List,
        target_sparsity: float,
        target_mask_size: int,
        device: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.mask_output_shape = mask_output_shape
        self.target_sparsity = target_sparsity

        self.droprate_init = 0.5
        self.temperature = 1.0 / 3.0
        self.magical_number = 0.8
        self.device = device

        self.z_loga = self.initialize_mask(mask_shape)
        self.mask_size = self.z_loga.shape[-1]
        self.target_mask_size = target_mask_size

    def param_init_fn(self, module):
        mean = 1.0
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)

    def initialize_mask(self, mask_shape: List):
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device))
        self.param_init_fn(z_loga)
        return z_loga

    def soft_saturation(self, mean=0.5):
        scale = 2.4 / mean
        z = ste_relu(self.z_loga)
        z = torch.sigmoid(scale * (z - mean))
        z = z * (limit_b - limit_a) + limit_a
        z = ste_clamp(z, min_val=0, max_val=1)
        return z

    def sample_z(self):
        z = ste_relu(self.z_loga).reshape(*self.mask_output_shape)
        return z

    def _deterministic_z(self, z_loga, uniform_sparsity):
        num_groups = self.z_loga.numel() // self.z_loga.size(-1)
        expected_num_zeros = (
            (self.mask_size - self.target_mask_size)
            if uniform_sparsity
            else (self.mask_size - self.target_mask_size) * num_groups
        )
        try:
            num_zeros = round(expected_num_zeros)
        except Exception:
            print("num of zeros is nan....")
            sys.exit()

        soft_mask = ste_relu(z_loga)

        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(self.z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.0
        return soft_mask

    def deterministic_z(self, uniform_sparsity):
        if self.z_loga.ndim == 1:
            z = self._deterministic_z(self.z_loga, uniform_sparsity).reshape(*self.mask_output_shape)
        else:
            if uniform_sparsity:
                z_loga = self.z_loga.reshape(-1, self.z_loga.shape[-1])
                z = []
                for i in range(z_loga.shape[0]):
                    z_ = self._deterministic_z(z_loga[i], uniform_sparsity)
                    z.append(z_)
                z = torch.stack(z)
            else:
                z = self._deterministic_z(self.z_loga.reshape(-1), uniform_sparsity)
            z = z.reshape(*self.mask_output_shape)
        return z

    def forward(self, uniform_sparsity):
        func = self.sample_z if self.training else self.deterministic_z
        z = func(uniform_sparsity).reshape(self.mask_output_shape)
        return z

    def calculate_expected_score_sparsity(self, mean):
        score = self.soft_saturation(mean=mean)
        sparsity = 1 - score.sum(-1) / self.mask_size
        return score, sparsity


class L0Module(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        l0_module_cfg = cfg

        self.pruning_modules = l0_module_cfg.pruning_modules
        self.device = device
        self.uniform_sparsity = l0_module_cfg.get("uniform_sparsity", True)
        self.num_layers = l0_module_cfg.num_layers
        self.num_experts = l0_module_cfg.get("num_experts", None)
        self.expert_intermediate_size = l0_module_cfg.get("expert_intermediate_size", None)
        self.num_attentention_head = l0_module_cfg.get("num_attentention_head", None)
        self.intermediate_size = l0_module_cfg.get("intermediate_size", None)

        self.masks = {}
        self.lambdas = {}
        self.target_sparsity = l0_module_cfg.target_sparsity
        for pruning_module in self.pruning_modules:
            self.initialize_one_module(pruning_module)
        self.masks = torch.nn.ModuleDict(self.masks)
        self.lambdas = torch.nn.ParameterDict(self.lambdas)

    def initialize_one_module(self, module_name: str):
        func_name = f"initialize_{module_name}"
        try:
            method = getattr(self, func_name)
        except AttributeError:
            raise NotImplementedError(f"Instance `{self}` does not implement `{func_name}`")
        method()

    def initialize_expert(self):
        mask_shape = [self.num_layers, self.num_experts, self.expert_intermediate_size]
        mask_output_shape = [self.num_layers, self.num_experts, self.expert_intermediate_size]

        target_int_sparsity = self.target_sparsity
        target_mask_size = math.ceil(self.expert_intermediate_size * (1 - target_int_sparsity))

        pd = {
            "lambda_1_expert": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
            "lambda_2_expert": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
            "lambda_3_expert": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
        }
        self.lambdas.update(pd)

        int_mask = Mask(
            name="expert",
            mask_shape=mask_shape,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_int_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
        )
        self.masks["expert"] = int_mask

    def initialize_intermediate(self):
        mask_shape = [self.num_layers, self.intermediate_size]
        mask_output_shape = [self.num_layers, self.intermediate_size]

        target_int_sparsity = self.target_sparsity
        target_mask_size = math.ceil(self.intermediate_size * (1 - target_int_sparsity))

        pd = {
            "lambda_1_intermediate": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
            "lambda_2_intermediate": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
            "lambda_3_intermediate": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
        }
        self.lambdas.update(pd)

        int_mask = Mask(
            name="intermediate",
            mask_shape=mask_shape,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_int_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
        )
        self.masks["intermediate"] = int_mask

    def initialize_head(self):
        mask_shape = [self.num_layers, self.num_attentention_head]
        mask_output_shape = [self.num_layers, self.num_attentention_head]

        target_int_sparsity = math.floor(self.target_sparsity * self.num_attentention_head) / self.num_attentention_head
        target_mask_size = math.ceil(self.num_attentention_head * (1 - target_int_sparsity))

        pd = {
            "lambda_1_head": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
            "lambda_2_head": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
            "lambda_3_head": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
        }
        self.lambdas.update(pd)

        int_mask = Mask(
            name="head",
            mask_shape=mask_shape,
            mask_output_shape=mask_output_shape,
            target_sparsity=target_int_sparsity,
            target_mask_size=target_mask_size,
            device=self.device,
        )
        self.masks["head"] = int_mask

    def calculate_expected_score_sparsity(self, mean):
        expected_scores = {}
        expected_sparsitys = {}
        for key in self.masks:
            score, sparsity = self.masks[key].calculate_expected_score_sparsity(mean=mean)
            expected_scores[key] = score
            expected_sparsitys[key] = sparsity
        return expected_scores, expected_sparsitys

    def lagrangian_regularization(self, progress):
        def _lag_loss(expected_sparsity: torch.Tensor, target_sparsity: float,
                      lambda_1: torch.Tensor, lambda_2: torch.Tensor):
            if not self.uniform_sparsity:
                expected_sparsity = expected_sparsity.mean()
            lag = lambda_1 * (expected_sparsity - target_sparsity) + lambda_2 * (expected_sparsity - target_sparsity) ** 2
            return lag.mean()

        def _binary_loss(score: torch.Tensor, lambda_3: torch.Tensor):
            return (lambda_3 * (1 - score) * score).mean()

        mean = 0.5 - (0.5 - 0.1) * (progress ** 0.5)
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity(mean=mean)
        lagrangian_loss = 0.0

        for pruning_module in self.pruning_modules:
            ts = self.masks[pruning_module].target_sparsity
            expected_ts = expected_sparsitys[pruning_module]

            lagrangian_loss = lagrangian_loss + _lag_loss(
                expected_ts, ts,
                self.lambdas[f"lambda_1_{pruning_module}"],
                self.lambdas[f"lambda_2_{pruning_module}"],
            )
            score = self.masks[pruning_module].soft_saturation(mean=mean)
            lagrangian_loss = lagrangian_loss + _binary_loss(score, self.lambdas[f"lambda_3_{pruning_module}"])

        return lagrangian_loss

    def forward(self, calculate_lagrangian: bool = False, progress: float = 0.0):
        if calculate_lagrangian:
            return self.lagrangian_regularization(progress)

        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}

        if self.training:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                zs[f"{pruning_module}_z"] = mask.sample_z()
        else:
            with torch.no_grad():
                for pruning_module in self.pruning_modules:
                    mask = self.masks[pruning_module]
                    zs[f"{pruning_module}_z"] = mask.deterministic_z(uniform_sparsity=self.uniform_sparsity)

        return zs


class WrappedModel(nn.Module):
    """
    Wrapper for base model + L0 masks.
    """
    def __init__(self, model, l0_config, distill: bool = False, ratio: float = 1.0, tau=1.0):
        super().__init__()
        self.base = model
        self.l0_module = L0Module(cfg=l0_config) if l0_config is not None else None
        self.distill = distill
        self.ratio = ratio
        self.tau = tau

    def distill_kl_loss(self, student_logits, teacher_logits, attn_mask=None, reduction="mean"):
        log_p_s = F.log_softmax(student_logits / self.tau, dim=-1)
        log_p_t = F.log_softmax(teacher_logits / self.tau, dim=-1)
        p_t = log_p_t.exp()
        kl_per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)

        if attn_mask is not None:
            kl_per_token = kl_per_token * attn_mask
            denom = attn_mask.sum().clamp_min(1)
        else:
            denom = kl_per_token.numel()

        if reduction == "mean":
            loss = kl_per_token.sum() / denom
        elif reduction == "sum":
            loss = kl_per_token.sum()
        else:
            loss = kl_per_token

        return loss * (self.tau ** 2)

    def _get_z_kwargs(self, progress: float):
        if self.l0_module is None:
            return {}
        z_dict = self.l0_module(calculate_lagrangian=False, progress=progress)
        z_kwargs = {}
        for key in ("expert_z", "head_z", "intermediate_z"):
            if key in z_dict and z_dict[key] is not None:
                z_kwargs[key] = z_dict[key]
        return z_kwargs

    def forward(self, input_ids=None, attention_mask=None, labels=None, progress: float = 0.0, **kwargs):
        z_kwargs = self._get_z_kwargs(progress=progress)

        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
            **z_kwargs,
            **kwargs,
        )

        lag_loss = 0.0
        if self.l0_module is not None:
            lag_loss = self.l0_module(calculate_lagrangian=True, progress=progress)

        output_v = {
            "train_loss": out.loss,
            "lag_loss": lag_loss if (lag_loss is not None and self.training) else 0.0,
        }

        if self.distill:
            # teacher run: no z kwargs
            with torch.no_grad():
                teacher_out = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=False,
                    return_dict=True,
                    **kwargs,
                )
            output_v["distill_loss"] = self.distill_kl_loss(out.logits, teacher_out.logits) * self.ratio

        return output_v


# -------------------------
# HF checks / logging
# -------------------------

check_min_version("4.57.0.dev0")
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# -------------------------
# Args
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task (torchrun+DDP)")

    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset name (datasets library).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="Dataset config name.")
    parser.add_argument("--validation_file", type=str, default=None, help="csv/txt/json validation file.")
    parser.add_argument("--validation_split_percentage", default=1, help="If no validation split exists.")
    parser.add_argument("--model_name_or_path", type=str, required=False, help="HF model path or name.")
    parser.add_argument("--config_name", type=str, default=None, help="HF config name/path.")
    parser.add_argument("--eval_steps", type=int, default=1200, help="Run eval every N optimizer steps.")
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)

    parser.add_argument("--z_learning_rate", type=float, default=5e-3)
    parser.add_argument("--lambda_13_learning_rate", type=float, default=2e-2)
    parser.add_argument("--lambda_2_learning_rate", type=float, default=4e-1)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--model_type", type=str, default=None, choices=MODEL_TYPES)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--no_keep_linebreaks", action="store_true")

    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--hub_model_id", type=str)
    parser.add_argument("--hub_token", type=str)

    parser.add_argument("--trust_remote_code", action="store_true")

    parser.add_argument("--checkpointing_steps", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--with_tracking", action="store_true")
    parser.add_argument("--report_to", type=str, default="all")

    parser.add_argument("--sparsity", action="store_true")
    parser.add_argument("--uniform_sparsity", action="store_true")
    parser.add_argument("--target_sparsity", type=float, default=0.50)
    parser.add_argument("--use_alpaca_format", action="store_true")
    parser.add_argument("--distill", action="store_true")
    parser.add_argument("--prunable_module", type=str, default="", help='Comma-separated modules e.g. "head,intermediate"')
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=1.0)

    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--expert_intermediate_size", type=int, default=1024)
    parser.add_argument("--num_attentention_head", type=int, default=1024)
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "no"],
        help="Autocast precision. bf16/fp16/no. fp16 uses GradScaler.",
    )
    parser.add_argument(
        "--tensorboard_logdir",
        type=str,
        default=None,
        help="TensorBoard log dir. Default: <output_dir>/tensorboard",
    )

    args = parser.parse_args()

    def get_extension(path: str) -> str:
        if path.endswith(".json.gz"):
            return "json"
        return path.split(".")[-1]

    
    if args.push_to_hub and args.output_dir is None:
        raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()

    rank, world_size, local_rank, device = ddp_init()
    is_main = (rank == 0)

    mp = args.mixed_precision
    use_cuda = (device.type == "cuda")

    @contextmanager
    def amp_ctx():
        if (not use_cuda) or (mp == "no"):
            yield
        else:
            dtype = torch.bfloat16 if mp == "bf16" else torch.float16
            with autocast(device_type="cuda", dtype=dtype):
                yield

    scaler = GradScaler(enabled=(use_cuda and mp == "fp16"))

    writer = None
    if args.with_tracking and (args.report_to == "all" or "tensorboard" in args.report_to):
        if is_main:
            tb_dir = args.tensorboard_logdir or os.path.join(args.output_dir or ".", "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            writer.add_text("config/args", json.dumps(vars(args), indent=2), global_step=0)

    def tb_scalar(tag: str, val: float, step: int):
        if writer is not None:
            writer.add_scalar(tag, float(val), step)

    # logging
    logging.basicConfig(
        format=f"%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.ERROR,
    )
    logger = logging.getLogger(__name__)

    print_rank0(is_main, f"[ddp] rank={rank} local_rank={local_rank} world_size={world_size} device={device}")

    if args.seed is not None:
        set_seed(args.seed, rank=rank)

    # Handle repo creation / output dir
    if is_main:
        if args.push_to_hub:
            repo_name = args.hub_model_id or Path(args.output_dir).absolute().name
            api = HfApi()
            _ = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            os.makedirs(args.output_dir, exist_ok=True)
            gitignore_path = os.path.join(args.output_dir, ".gitignore")
            content = ""
            if os.path.exists(gitignore_path):
                with open(gitignore_path, "r") as f:
                    content = f.read()
            with open(gitignore_path, "a") as f:
                if content and not content.endswith("\n"):
                    f.write("\n")
                if "step_*" not in content:
                    f.write("step_*\n")
                if "epoch_*" not in content:
                    f.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    ddp_barrier()

    # Build L0 config
    modules = [m for m in args.prunable_module.split(",") if m]
    l0_config = L0Config(
        pruning_modules=modules,
        target_sparsity=args.target_sparsity,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        intermediate_size=args.intermediate_size,
        expert_intermediate_size=args.expert_intermediate_size,
        num_attentention_head=args.num_attentention_head,
        uniform_sparsity=args.uniform_sparsity,
    )
    print_rank0(is_main, l0_config)
    print_rank0(is_main, args)

    

    # -------------------------
    # Model / tokenizer
    # -------------------------
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,     # changed
            low_cpu_mem_usage=True,         # changed
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)



    if args.sparsity:
        model = WrappedModel(model, l0_config=l0_config, distill=args.distill, ratio=args.ratio, tau=args.tau)

    
    base_model = model.base if isinstance(model, WrappedModel) else model

    # enable gradient checkpoint (your flag)
    if args.gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.config.use_cache = False




    from datasets import load_from_disk
    print(args.dataset_name)
    lm_datasets = load_from_disk(args.dataset_name)


    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    if is_main and len(train_dataset) > 0:
        idx = random.randrange(len(train_dataset))
        logger.info(f"Sample {idx} of the training set: {train_dataset[idx]}.")

    # -------------------------
    # Dataloaders (DistributedSampler)
    # -------------------------
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        shuffle=False,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # -------------------------
    # Optimizer (your grouping)
    # -------------------------
    z_loga_name = "z_loga"
    lambda_name = "lambda"
    other_lambda_names = ["lambda_1", "lambda_3"]

    z_loga_params = []
    lambda_params = []
    lambda_other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if z_loga_name in n:
            z_loga_params.append(p)
        elif lambda_name in n:
            if any(x in n for x in other_lambda_names):
                lambda_other_params.append(p)
            else:
                lambda_params.append(p)

    optimizer_grouped_parameters = [
        {"params": z_loga_params, "lr": args.z_learning_rate, "weight_decay": 0.0},
        {"params": lambda_params, "lr": args.lambda_2_learning_rate, "weight_decay": 0.0, "maximize": True},
        {"params": lambda_other_params, "lr": args.lambda_13_learning_rate, "weight_decay": 0.0, "maximize": True},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # freeze everything, then enable only l0_module trainables (as you did)
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "l0_module" in n:
            p.requires_grad = True

    # -------------------------
    # Scheduler / steps
    # (keeps your "world_size scaling" behavior)
    # -------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * world_size,
        num_training_steps=(args.max_train_steps if overrode_max_train_steps else args.max_train_steps * world_size),
    )

    # -------------------------
    # DDP wrap
    # -------------------------
    model = model.to(device)
    if world_size > 1 and device.type == "cuda":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    elif world_size > 1:
        model = DDP(model)

    # -------------------------
    # Resume (trainable params only)
    # -------------------------
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    completed_steps = 0
    starting_epoch = 0
    resume_step = None

    def load_trainable_params(folder: str):
        pt = os.path.join(folder, "trainable_params.pt")
        if not os.path.exists(pt):
            return False
        sd = torch.load(pt, map_location="cpu")
        m = unwrap_ddp(model)
        missing = []
        with torch.no_grad():
            for n, p in m.named_parameters():
                if p.requires_grad and n in sd:
                    p.copy_(sd[n].to(p.device, dtype=p.dtype))
                elif p.requires_grad and n not in sd:
                    missing.append(n)
        if is_main and missing:
            logger.warning(f"[resume] missing trainable params: {missing[:10]}{'...' if len(missing)>10 else ''}")
        return True

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        path = os.path.basename(checkpoint_path.rstrip("/"))
        ok = load_trainable_params(checkpoint_path)
        print_rank0(is_main, f"[resume] loaded trainable_params.pt: {ok} from {checkpoint_path}")

        training_difference = os.path.splitext(path)[0]
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        elif "step" in training_difference:
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    ddp_barrier()

    # -------------------------
    # Eval
    # -------------------------
    def run_eval(model_ddp, eval_dl, completed_steps, progress):
        model_ddp.eval()
        loss_sum = torch.tensor(0.0, device=device)
        tok_count = torch.tensor(0, device=device, dtype=torch.long)

        for batch in eval_dl:
            batch = move_to_device(batch, device)
            with torch.no_grad():
                with amp_ctx():
                    outputs = model_ddp(**batch, use_cache=False, past_key_values=None, progress=progress)

            if torch.is_tensor(outputs):
                loss = outputs
            elif isinstance(outputs, dict):
                loss = outputs["train_loss"]
            else:
                loss = outputs.loss

            labels = batch["labels"]
            n_tok = (labels != -100).sum()

            loss_sum += loss.detach() * n_tok
            tok_count += n_tok

        ddp_all_reduce_(loss_sum, op=dist.ReduceOp.SUM)
        ddp_all_reduce_(tok_count, op=dist.ReduceOp.SUM)

        eval_loss = (loss_sum / tok_count.clamp_min(1)).float()
        perplexity = float("inf") if eval_loss.item() > 100 else math.exp(eval_loss.item())

        if is_main:
            tb_scalar("eval/eval_loss", eval_loss.item(), completed_steps)
            tb_scalar("eval/perplexity", perplexity, completed_steps)

        if is_main:
            logger.info(f"[eval] step {completed_steps}: ppl={perplexity:.4f} eval_loss={eval_loss.item():.6f}")

        model_ddp.train()
        return eval_loss, perplexity

    # recompute epochs if max_train_steps was set weirdly
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps

    if is_main:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not is_main)
    if completed_steps > 0:
        progress_bar.update(completed_steps)

    # -------------------------
    # Train loop (native grad accumulation + DDP no_sync)
    # -------------------------
    grad_accum = args.gradient_accumulation_steps

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # skip batches when resuming within the starting epoch
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None and step < resume_step:
                continue

            if completed_steps >= args.max_train_steps:
                break

            batch = move_to_device(batch, device)
            is_accumulating = ((step + 1) % grad_accum != 0)

            sync_ctx = model.no_sync() if (world_size > 1 and is_accumulating) else nullcontext()

            progress = (completed_steps + 0.0) / max(args.max_train_steps, 1)

            with sync_ctx:
                with amp_ctx():
                    outputs = model(**batch, use_cache=False, past_key_values=None, progress=progress)


                if torch.is_tensor(outputs):
                    train_loss = outputs
                    lag_loss = torch.tensor(0.0, device=device)
                    distill_loss = torch.tensor(0.0, device=device)
                    loss = outputs
                elif isinstance(outputs, dict):
                    train_loss = outputs["train_loss"]
                    lag_loss = outputs.get("lag_loss", torch.tensor(0.0, device=device))
                    distill_loss = outputs.get("distill_loss", torch.tensor(0.0, device=device))
                    loss = train_loss + lag_loss + (distill_loss if args.distill else 0.0)
                else:
                    train_loss = outputs.loss
                    lag_loss = torch.tensor(0.0, device=device)
                    distill_loss = torch.tensor(0.0, device=device)
                    loss = outputs.loss

                loss_to_backprop = loss / grad_accum
                if scaler.is_enabled():
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

            if not is_accumulating:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                completed_steps += 1
                progress_bar.update(1)

                # reduce for logging
                t_train = ddp_reduce_mean_scalar(train_loss.detach().float(), world_size)
                t_lag = ddp_reduce_mean_scalar(lag_loss.detach().float(), world_size)
                if args.distill:
                    t_distill = ddp_reduce_mean_scalar(distill_loss.detach().float(), world_size)

                if is_main:
                    tb_scalar("train/train_loss", t_train.item(), completed_steps)
                    tb_scalar("train/lag_loss", t_lag.item(), completed_steps)
                    if args.distill:
                        tb_scalar("train/distill_loss", t_distill.item(), completed_steps)
                    tb_scalar("train/lr", lr_scheduler.get_last_lr()[0], completed_steps)
                    if scaler.is_enabled():
                        tb_scalar("train/grad_scale", scaler.get_scale(), completed_steps)

                # clamp after update (on unwrapped)
                with torch.no_grad():
                    m = unwrap_ddp(model)
                    for n, p in m.named_parameters():
                        if p.requires_grad and "z_loga" in n:
                            p.clamp_(min=-0.1)

                if is_main:
                    logger.info(f"step {completed_steps}: train_loss {t_train.item():.6f}")
                    logger.info(f"step {completed_steps}: lag_loss   {t_lag.item():.6f}")
                    if args.distill:
                        logger.info(f"step {completed_steps}: distill_loss {t_distill.item():.6f}")

                # if args.eval_steps is not None and completed_steps % args.eval_steps == 0 and completed_steps > 0:
                #     run_eval(model, eval_dataloader, completed_steps, progress)

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0 and completed_steps > 0:
                        out_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            out_dir = os.path.join(args.output_dir, out_dir)
                        if is_main:
                            os.makedirs(out_dir, exist_ok=True)
                            m = unwrap_ddp(model)
                            d = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
                            torch.save(d, os.path.join(out_dir, "trainable_params.pt"))

        if completed_steps >= args.max_train_steps:
            break

    ddp_barrier()
    if writer is not None:
        writer.flush()
        writer.close()


    if ddp_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
