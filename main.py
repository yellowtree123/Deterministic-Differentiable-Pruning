#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers @ git+https://github.com/huggingface/transformers.git",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast
import transformers
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
import torch.nn as nn
import sys
from dataclasses import dataclass, field
from typing import List, Any, Optional, List
import torch.nn.functional as F
@dataclass
class L0Config:
    # required-ish fields your code uses
    pruning_modules: List[str] = field(default_factory=list)  # e.g. ["head", "intermediate"]
    target_sparsity: float = 0.20
    num_layers: int = 16
    num_experts: int = 64
    intermediate_size: int = 1024
    expert_intermediate_size : int = 1024
    num_attentention_head : int = 1024

    # optional fields accessed via .get(...)
    uniform_sparsity: bool = False

    # dict-like helper (so l0_module_cfg.get("x", default) works)
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)



limit_a, limit_b, epsilon = -.1, 1.1, 1e-6   

def ste_clamp(x: torch.Tensor, min_val: float = 0.0, max_val: float = 1.0) -> torch.Tensor:
    """
    Straight-Through Estimator clamp:
      forward: clamp(x, min_val, max_val)
      backward: gradient w.r.t. x is identity (no clamp).
    """
    x_hard = torch.clamp(x, min_val, max_val)
    return x + (x_hard - x).detach()

def ste_relu(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through ReLU:
      forward: relu(x)
      backward: gradient ≈ 1 everywhere (like identity)
    """
    x_hard = torch.relu(x)           # forward: 0 for x<0, x for x>=0
    return x + (x_hard - x).detach() # backward: ∂/∂x ≈ 1

class Mask(nn.Module):
    def __init__(self, 
                 name: str,
                 mask_shape: List, 
                 mask_output_shape: List, 
                 target_sparsity: float,
                 target_mask_size: int,
                 device: str
                ) -> None:
        super().__init__()
        self.name = name
        self.mask_output_shape = mask_output_shape
        self.target_sparsity=target_sparsity

        self.droprate_init = 0.5
        self.temperature = 1./3.
        self.magical_number = 0.8
        self.device = device
        
        self.z_loga = self.initialize_mask(mask_shape) 
        self.mask_size = self.z_loga.shape[-1] # the full size of each unit
        self.target_mask_size = target_mask_size
        
        
    def param_init_fn(self, module):
        """ Initialize the parameters for masking variables. """
        mean = math.log(1 - self.droprate_init) - math.log(self.droprate_init)
        mean = 1.0
        if isinstance(module, nn.Parameter):
            module.data.normal_(mean, 1e-2)
        else:
            for tensor in module.parameters():
                tensor.data.normal_(mean, 1e-2)
        
    def initialize_mask(self, mask_shape: List):
        """ Initialize the parameters for masking variables. """
        z_loga = nn.Parameter(torch.ones(*mask_shape, device=self.device))
        self.param_init_fn(z_loga)
        return z_loga
    
    
    def soft_saturation(self, mean=0.5):
        scale = 2.4/mean
        z = ste_relu(self.z_loga)
        z =  torch.sigmoid( scale * (z-mean))
        z = z * (limit_b - limit_a) + limit_a
        z = ste_clamp(z, min_val=0, max_val=1)
        return z
        

    
    def sample_z(self):
        z = ste_relu(self.z_loga).reshape(*self.mask_output_shape)
        return z
    
    def _deterministic_z(self, z_loga, uniform_sparsity):
        # Following https://github.com/asappresearch/flop/blob/e80e47155de83abbe7d90190e00d30bfb85c18d5/flop/hardconcrete.py#L8 line 103
        num_groups = self.z_loga.numel() // self.z_loga.size(-1)
        expected_num_zeros = (self.mask_size - self.target_mask_size)  if uniform_sparsity else (self.mask_size - self.target_mask_size) * num_groups
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            sys.exit()

        soft_mask = ste_relu(z_loga)

        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(self.z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
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
        super(L0Module, self).__init__()


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
            raise NotImplementedError("Instance `{}` does not implement `{}`".format(self, func_name))
        method()

    def initialize_expert(self):
        mask_shape = [self.num_layers, self.num_experts, self.expert_intermediate_size]
        mask_output_shape = [self.num_layers, self.num_experts, self.expert_intermediate_size] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 

        target_int_sparsity =  self.target_sparsity
        target_mask_size =  math.ceil(self.expert_intermediate_size * (1-target_int_sparsity))
        pd = {"lambda_1_expert": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
                "lambda_2_expert": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
                "lambda_3_expert": torch.nn.Parameter(torch.tensor([0.0], device=self.device))}
        self.lambdas.update(pd)
        
        int_mask = Mask(name="expert",
                        mask_shape=mask_shape,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                        device=self.device)
        self.masks["expert"] = int_mask

    def initialize_intermediate(self):
        mask_shape = [self.num_layers, self.intermediate_size]
        mask_output_shape = [self.num_layers, self.intermediate_size] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 

        target_int_sparsity =  self.target_sparsity
        target_mask_size =  math.ceil(self.intermediate_size * (1-target_int_sparsity))
        pd = {"lambda_1_intermediate": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
                "lambda_2_intermediate": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
                "lambda_3_intermediate": torch.nn.Parameter(torch.tensor([0.0], device=self.device))}
        self.lambdas.update(pd)
        
        int_mask = Mask(name="intermediate",
                        mask_shape=mask_shape,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                        device=self.device)
        self.masks["intermediate"] = int_mask

    def initialize_head(self):
        mask_shape = [self.num_layers, self.num_attentention_head]
        mask_output_shape = [self.num_layers, self.num_attentention_head] 
        
        target_int_sparsity = None; pd = {}; target_mask_size=None; 

        target_int_sparsity =  math.floor(self.target_sparsity * self.num_attentention_head) / self.num_attentention_head
        target_mask_size =  math.ceil(self.num_attentention_head * (1-target_int_sparsity))

        pd = {"lambda_1_head": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
                "lambda_2_head": torch.nn.Parameter(torch.tensor([0.0], device=self.device)),
                "lambda_3_head": torch.nn.Parameter(torch.tensor([0.0], device=self.device))}
        self.lambdas.update(pd)
        
        int_mask = Mask(name="head",
                        mask_shape=mask_shape,
                        mask_output_shape=mask_output_shape,
                        target_sparsity=target_int_sparsity,
                        target_mask_size=target_mask_size,
                        device=self.device)
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
        def _lag_loss(expected_sparsity: torch.tensor, target_sparsity: float, lambda_1: torch.tensor, lambda_2: torch.tensor, abs: bool = True):

            if not self.uniform_sparsity:
                expected_sparsity = expected_sparsity.mean()

            lagrangian_loss = lambda_1 * (expected_sparsity - target_sparsity) + lambda_2 * (expected_sparsity - target_sparsity) ** 2 
            lagrangian_loss = lagrangian_loss.mean()
            return lagrangian_loss

        def _binary_loss(score: torch.tensor, lambda_3: torch.tensor):
            binary_loss = lambda_3 * (1 - score) * score
            binary_loss = binary_loss.mean()
            return binary_loss     

        mean = 0.5 - (0.5 - 0.1) * (progress ** 0.5)
        expected_scores, expected_sparsitys = self.calculate_expected_score_sparsity(mean=mean)    
        lagrangian_loss = 0
        return_v = {}
        for pruning_module in self.pruning_modules:
            ts = self.masks[pruning_module].target_sparsity
            expected_ts = expected_sparsitys[pruning_module] 

            lagrangian_loss += _lag_loss(expected_ts, ts, self.lambdas[f"lambda_1_{pruning_module}"], self.lambdas[f"lambda_2_{pruning_module}"])
            score = self.masks[pruning_module].soft_saturation(mean=mean)
            lagrangian_loss += _binary_loss(score, self.lambdas[f"lambda_3_{pruning_module}"])
            expected_ts = expected_ts.mean()
            return_v.update({"expected_{}_sparsity".format(pruning_module): expected_ts, "target_{}_sparsity".format(pruning_module): ts})


        # return_v might not matter
        return lagrangian_loss
 
    def forward(self, calculate_lagrangian: bool = False, progress: float = 0.0):
        if calculate_lagrangian:
            return self.lagrangian_regularization(progress)
        
        zs = {f"{pruning_module}_z": [] for pruning_module in self.pruning_modules}
                
        if self.training:
            for pruning_module in self.pruning_modules:
                mask = self.masks[pruning_module]
                z = mask.sample_z()
                zs[f"{pruning_module}_z"] = z
        else: # removed layerwise! 
            with torch.no_grad():
                for pruning_module in self.pruning_modules:
                    mask = self.masks[pruning_module]
                    z = mask.deterministic_z(uniform_sparsity=self.uniform_sparsity)
                    zs[f"{pruning_module}_z"] = z
        return zs 


import torch.nn as nn
import torch.nn.functional as F

class WrappedModel(nn.Module):
    """
    Unified wrapper for models that may accept any of:
      - expert_z (e.g., OLMoE)
      - head_z, intermediate_z (e.g., LLaMA)
    It forwards whichever z tensors L0Module returns.
    """
    def __init__(self, model, l0_config, distill: bool = False, ratio: float = 1.0, tau=1.0):
        super().__init__()
        self.base = model
        self.l0_module = L0Module(cfg=l0_config) if l0_config is not None else None
        self.distill = distill
        self.ratio = ratio
        self.tau = tau

    def distill_kl_loss(self, student_logits, teacher_logits, attn_mask=None, reduction="mean"):
        # log p_s and p_t at temperature tau
        log_p_s = F.log_softmax(student_logits / self.tau, dim=-1)   # (B,T,V)
        log_p_t = F.log_softmax(teacher_logits / self.tau, dim=-1)   # (B,T,V)
        p_t = log_p_t.exp()                                     # (B,T,V)

        kl_per_token = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)  # (B,T)

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
            loss = kl_per_token  # (B,T)

        return loss * (self.tau ** 2)

    def _get_z_kwargs(self, progress: float):
        """
        Query L0Module once and return kwargs to pass into the base model,
        containing only supported z's present in the returned dict.
        """
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
            output_hidden_states=True,
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
            # Teacher run should be "unpruned": no z kwargs
            with torch.no_grad():
                teacher_out = self.base(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True,
                    **kwargs,
                )
            output_v["distill_loss"] = self.distill_kl_loss(out.logits, teacher_out.logits) * self.ratio

        return output_v


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=500, 
        help="Run evaluation every N optimizer steps."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--z_learning_rate",
        type=float,
        default=5e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lambda_13_learning_rate",
        type=float,
        default=2e-2,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lambda_2_learning_rate",
        type=float,
        default=4e-1,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--sparsity",
        action="store_true",
        help=(
            "Whether or not to apply l0 module."
        ),
    )
    parser.add_argument(
        "--uniform_sparsity",
        action="store_true",
        help=(
            "Whether or not to use uniform sparsity among experts."
        ),
    )
    parser.add_argument(
        "--target_sparsity",
        type=float,
        default=0.50,
        help=(
            "Total sparsity among all experts."
        ),
    )
    parser.add_argument(
        "--use_alpaca_format",
        action="store_true",
        help=(
            "Whether or not to use uniform sparsity among experts."
        ),
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help=(
            "Whether or not to use distillation."
        ),
    )
    parser.add_argument(
        "--enable_gradient_checkpointing",
        action="store_true",
        help=(
            "Whether or not to use gradient checkpointing."
        ),
    )
    parser.add_argument(
        "--prunable_module",
        type=str,
        default="",
        help='Comma-separated modules, e.g. "head,intermediate"'
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=1.0,
        help=(
            "Distillation loss coefficient"
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help=(
            "Distillation tau coefficient"
        ),
    )
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--num_experts", type=int, default=64)
    parser.add_argument("--intermediate_size", type=int, default=1024)
    parser.add_argument("--expert_intermediate_size", type=int, default=1024)
    parser.add_argument("--num_attentention_head", type=int, default=1024)
    args = parser.parse_args()



    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args


def main():
    args = parse_args()
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
    print(l0_config)
    print(args)
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

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
    accelerator.wait_for_everyone()

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16,     
            low_cpu_mem_usage=True,         
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    if args.sparsity:
        model = WrappedModel(model, l0_config=l0_config, distill=args.distill, ratio=args.ratio, tau=args.tau)
    base_model = model.base if hasattr(model, "base") else model

    if args.enable_gradient_checkpointing:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.config.use_cache = False

    for p in model.parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if "l0_module" in n:   # naming depends on your implementation/PEFT
            p.requires_grad = True

    from datasets import load_from_disk
    print(args.dataset_name)
    lm_datasets = load_from_disk(args.dataset_name)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    l0_name = "l0_module"
    z_loga_name = "z_loga"
    lambda_name = "lambda"

    # add whatever extra lambdas you want to break out
    other_lambda_names = ["lambda_1", "lambda_3"]  # extend as needed

    z_loga_params = []
    lambda_params = []
    lambda_other_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if z_loga_name in n:
            z_loga_params.append(p)

        # lambdas
        elif lambda_name in n:
            if any(x in n for x in other_lambda_names):
                lambda_other_params.append(p)
            else:
                lambda_params.append(p)


    optimizer_grouped_parameters = [
        {
            "params": z_loga_params,
            "lr": args.z_learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": lambda_params,
            "lr": args.lambda_2_learning_rate,
            "weight_decay": args.weight_decay,
            "maximize": True,
        },
        {
            "params": lambda_other_params,
            "lr": args.lambda_13_learning_rate,   # you can use args.lambda2_lr etc. if you want
            "weight_decay": args.weight_decay,
            "maximize": True,
        },
    ]



    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)




    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )
    # Prepare everything with our `accelerator`.
    optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    from torch.nn.parallel import DistributedDataParallel as DDP
    model= model.cuda()
    model = DDP(model)

    def run_eval(model, eval_dataloader, completed_steps, progress, max_eval_batches=100):
        model.eval()

        loss_sum = torch.tensor(0.0, device=accelerator.device)
        tok_count = torch.tensor(0, device=accelerator.device, dtype=torch.long)

        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx >= max_eval_batches:
                break

            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(
                        **batch,
                        use_cache=False,
                        past_key_values=None,
                        progress=progress,
                    )

            if torch.is_tensor(outputs):
                loss = outputs
            elif isinstance(outputs, dict):
                loss = outputs.get("train_loss", outputs.get("loss"))
                if loss is None:
                    raise ValueError("Could not find eval loss in outputs dict.")
            else:
                loss = outputs.loss

            labels = batch["labels"]
            n_tok = (labels != -100).sum()

            loss_sum += loss.detach() * n_tok
            tok_count += n_tok

        # sum across GPUs
        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        tok_count = accelerator.reduce(tok_count, reduction="sum")

        eval_loss = (loss_sum / tok_count).float()
        perplexity = float("inf") if eval_loss.item() > 100 else math.exp(eval_loss.item())

        if accelerator.is_main_process:
            logger.info(
                f"step {completed_steps}: perplexity: {perplexity} eval_loss: {eval_loss.item():.6f}"
            )

        if args.with_tracking:
            accelerator.log(
                {"eval_loss": eval_loss.item(), "perplexity": perplexity},
                step=completed_steps,
            )

        model.train()

        return eval_loss, perplexity


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0


    for epoch in range(args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader

        train_loss_accum = torch.tensor(0.0, device=accelerator.device)
        train_loss_count = 0
        lag_loss_accum = torch.tensor(0.0, device=accelerator.device)
        lag_loss_count = 0
        if args.distill:
            distill_loss_accum = torch.tensor(0.0, device=accelerator.device)
            distill_loss_count = 0

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                progress = (completed_steps+0.0) / args.max_train_steps
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(**batch, use_cache=False, past_key_values=None, progress=progress)
                if torch.is_tensor(outputs):
                    loss = outputs
                elif isinstance(outputs, dict):
                    train_loss = outputs["train_loss"] 
                    lag_loss = outputs["lag_loss"]
                    if args.distill:
                        distill_loss = outputs["distill_loss"]
                        loss = train_loss + lag_loss + distill_loss
                    else:
                        loss = train_loss + lag_loss 
                else:
                    loss = outputs.loss


                # accumulate microstep loss
                train_loss_accum += train_loss.detach()
                train_loss_count += 1
                lag_loss_accum += lag_loss.detach()
                lag_loss_count += 1    
                if args.distill:   
                    distill_loss_accum += distill_loss.detach()
                    distill_loss_count += 1         


                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)                     
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                train_step_loss = train_loss_accum / max(train_loss_count, 1)
                train_step_loss = accelerator.reduce(train_step_loss, reduction="mean")  # avg across GPUs
                lag_step_loss = lag_loss_accum / max(lag_loss_count, 1)
                lag_step_loss = accelerator.reduce(lag_step_loss, reduction="mean")  # avg across GPUs
                if args.distill:
                    distill_step_loss = distill_loss_accum / max(distill_loss_count, 1)
                    distill_step_loss = accelerator.reduce(distill_step_loss, reduction="mean")  # avg across GPUs
                # ✅ Clamp params AFTER update
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if not p.requires_grad:
                            continue
                        if "z_loga" in n:
                            p.clamp_(min=-0.1)  

                if accelerator.is_main_process:
                    logger.info(f"step {completed_steps}: train_loss {train_step_loss.item():.6f}")
                    logger.info(f"step {completed_steps}: lag_loss {lag_step_loss.item():.6f}")
                    if args.distill:
                        logger.info(f"step {completed_steps}: distill_loss {distill_step_loss.item():.6f}")

                if args.with_tracking:
                    accelerator.log(
                        {"train_loss_step": train_step_loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
                        step=completed_steps,
                    )
                    accelerator.log(
                        {"lag_loss_step": lag_step_loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
                        step=completed_steps,
                    )
                    if args.distill:
                        accelerator.log(
                            {"distill_loss_step": distill_step_loss.item(), "lr": lr_scheduler.get_last_lr()[0]},
                            step=completed_steps,
                        )                        

                # reset for next optimizer step
                train_loss_accum.zero_()
                train_loss_count = 0
                lag_loss_accum.zero_()
                lag_loss_count = 0
                if args.distill:
                    distill_loss_accum.zero_()
                    distill_loss_count = 0

                if args.eval_steps is not None and completed_steps % args.eval_steps == 0 and completed_steps>0:
                    progress = (completed_steps+0.0) / args.max_train_steps
                    run_eval(model, eval_dataloader, completed_steps, progress)

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients and completed_steps>0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    d = {}
                    for n, p in model.named_parameters():
                        if p.requires_grad:
                            d[n] = p.detach().cpu()
                    os.makedirs(output_dir, exist_ok=True)
                    torch.save(d, os.path.join(output_dir, "trainable_params.pt"))
            if completed_steps >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
