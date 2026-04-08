import math

import torch
import transformers
from transformers import AutoTokenizer, GenerationConfig
from utils.process_args import process_args
from utils.utils import set_seed
import pprint
import os
from datasets import load_dataset
import torch.nn as nn
import torch
import pickle



def lightning_prune_qwen_expert(model):

    import torch.nn.functional as F
    import re
    import torch    
    sd = torch.load("trainable_params.pt")
    expert_z = sd['module.l0_module.masks.expert.z_loga']
    expert_z = F.relu(expert_z) 


    # import math
    # limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
    # expert_z=  torch.sigmoid( expert_z *3)
    # expert_z = expert_z * (limit_b - limit_a) + limit_a
    # expert_z= torch.clamp(expert_z, min=0, max=1)

    pat1 = re.compile(r"layers\.(\d+)\.mlp\.experts")
    pat2 = re.compile(r"\.experts\.(\d+)")

    for name, module in model.named_modules():


        # Identify the experts container
        if not (hasattr(module, "gate_proj") and hasattr(module, "down_proj")):
            continue
        m = pat1.search(name)
        if m is None:
            continue
        layer_num = int(m.group(1))

        n = pat2.search(name)
        if n is None:
            continue
        expert_num = int(n.group(1))


        down = module.down_proj
        z = expert_z[layer_num,expert_num].to(device=down.weight.device, dtype=down.weight.dtype)

        # print(z.shape)
        # print(down.weight.shape)
        # print(name)
        with torch.no_grad():    
            down.weight.mul_(z)           # broadcast to [E, 1, I]

def lightning_prune_qwen_dense(model, mask_path):

    import torch.nn.functional as F
    import re
    import torch    
    sd = torch.load(mask_path)
    head_z = sd['l0_module.masks.head.z_loga'].to(torch.bfloat16)
    intermediate_z = sd['l0_module.masks.intermediate.z_loga'].to(torch.bfloat16)
    head_z = F.relu(head_z) 
    intermediate_z = F.relu(intermediate_z) 

    print( (head_z>0).sum() / head_z.numel() )
    print( (intermediate_z>0).sum() / intermediate_z.numel() )




    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP



    with torch.no_grad():
        for name, module in model.named_modules():


            if isinstance(module, Qwen3Attention):
                layer_num = int(re.search(r"layers\.(\d+)\.self_attn", name).group(1))
                mask = head_z[layer_num,:].to(module.v_proj.weight.device)
                head_z_for_update = torch.repeat_interleave(mask, module.head_dim)
                # module.v_proj.weight.data = module.v_proj.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
                module.o_proj.weight.mul_(head_z_for_update.unsqueeze(0))

            elif isinstance(module, Qwen3MLP):
                layer_num = int(re.search(r"layers\.(\d+)\.mlp", name).group(1))
                mask = torch.tensor(intermediate_z[layer_num,:]).to(module.up_proj.weight.device)
                module.up_proj.weight.data  = module.up_proj.weight.data.transpose(0, 1).mul(mask).transpose(0, 1)



def lightning_prune_qwen_dense_hc(model, mask_path):

    import torch.nn.functional as F
    import re
    import torch    
    sd = torch.load(mask_path)
    head_mask = sd['l0_module.masks.head.z_loga'].to(torch.bfloat16)
    intermediate_mask = sd['l0_module.masks.intermediate.z_loga'].to(torch.bfloat16)


    temperature = 1.0 / 3.0
    limit_a, limit_b, epsilon = -0.1, 1.1, 1e-6

    xn = (0 - limit_a) / (limit_b - limit_a)
    logits = math.log(xn) - math.log(1 - xn)
    head_z= torch.sigmoid(logits * temperature - head_mask).clamp(min=epsilon, max=1 - epsilon)
    head_z = 1 - head_z

    intermediate_z= torch.sigmoid(logits * temperature - intermediate_mask).clamp(min=epsilon, max=1 - epsilon)
    intermediate_z = 1 - intermediate_z


    k = int(head_z.numel() * 0.2)
    threshold = torch.topk(head_z.view(-1), k, largest=False).values.max()
    head_z = torch.where(head_z < threshold, torch.zeros_like(head_z), head_z)

    k = int(intermediate_z.numel() * 0.2)
    threshold = torch.topk(intermediate_z.view(-1), k, largest=False).values.max()
    intermediate_z = torch.where(intermediate_z < threshold, torch.zeros_like(intermediate_z), intermediate_z)





    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP



    with torch.no_grad():
        for name, module in model.named_modules():


            if isinstance(module, Qwen3Attention):
                layer_num = int(re.search(r"layers\.(\d+)\.self_attn", name).group(1))
                mask = head_z[layer_num,:].to(module.v_proj.weight.device)
                head_z_for_update = torch.repeat_interleave(mask, module.head_dim)
                # module.v_proj.weight.data = module.v_proj.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)
                module.o_proj.weight.mul_(head_z_for_update.unsqueeze(0))

            elif isinstance(module, Qwen3MLP):
                layer_num = int(re.search(r"layers\.(\d+)\.mlp", name).group(1))
                mask = torch.tensor(intermediate_z[layer_num,:]).to(module.up_proj.weight.device)
                module.up_proj.weight.data  = module.up_proj.weight.data.transpose(0, 1).mul(mask).transpose(0, 1)


def lightning_prune_llama_dense(model, mask_path):

    import torch.nn.functional as F
    import re
    import torch    
    print("Loading mask from ", mask_path)
    sd = torch.load(mask_path)
    head_z = sd['l0_module.masks.head.z_loga'].to(torch.bfloat16)
    intermediate_z = sd['l0_module.masks.intermediate.z_loga'].to(torch.bfloat16)
    head_z = F.relu(head_z) 
    intermediate_z = F.relu(intermediate_z) 

    print( (head_z>0).sum() / head_z.numel() )
    print( (intermediate_z>0).sum() / intermediate_z.numel() )



    import re
    import torch

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP




    for name, module in model.named_modules():


        if isinstance(module, LlamaAttention):
            layer_num = int(re.search(r"layers\.(\d+)\.self_attn", name).group(1))
            mask = head_z[layer_num,:].to(module.v_proj.weight.device)
            head_z_for_update = torch.repeat_interleave(mask, module.head_dim)
            module.v_proj.weight.data = module.v_proj.weight.data.transpose(0, 1).mul(head_z_for_update).transpose(0, 1)

        elif isinstance(module, LlamaMLP):
            layer_num = int(re.search(r"layers\.(\d+)\.mlp", name).group(1))
            mask = torch.tensor(intermediate_z[layer_num,:]).to(module.up_proj.weight.device)
            module.up_proj.weight.data = module.up_proj.weight.data = module.up_proj.weight.data.transpose(0, 1).mul(mask).transpose(0, 1)



@torch.no_grad()
def eval_ppl(model, bs=2, device="cuda", block_size=1024, model_path=None):
    testdata = load_dataset('/root/autodl-tmp/evaluation/wikitext-2-raw-v1', split='test',cache_dir=None)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True,)
    testenc = tokenizer.encode("\n\n".join(testdata['text']))
    model.eval()
    # transfrom list to tensor
    testenc = torch.tensor(testenc, dtype=torch.long, device=device).unsqueeze(0)
    # Calculate number of samples
    nsamples = testenc.numel() // block_size

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * block_size):(j * block_size)].to(device)
        inputs = inputs.reshape(j-i, block_size)

        # Forward pass through the model
        lm_logits = model(inputs)[0]

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * block_size * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * block_size))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    model.train()
    return ppl.item()



def main():
    args, logger = process_args()
    logger.info('Arguments: ')
    logger.info(pprint.pformat(vars(args)))
    logger.info('--' * 30)
    set_seed(args.seed)
    logger.info("seed set to {}".format(args.seed))

    config = transformers.AutoConfig.from_pretrained(
        args.model_path, token=None, trust_remote_code=True
    )

    logger.info(config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    from transformers import  AutoModelForCausalLM
    model =  AutoModelForCausalLM.from_pretrained(args.model_path, config=config, torch_dtype="auto")                                               


    model.use_cache = False

    # pruning_global(model, config, args, logger)
    # lightning_prune(model)
    device = args.device
    # lightning_prune_qwen_dense(model, args.mask_path)
    # lightning_prune_qwen_dense_hc(model, args.mask_path)
    lightning_prune_llama_dense(model, args.mask_path)
    model = model.to(device)
    model.half()
    model.config.use_cache = False
    with torch.no_grad():
        print(eval_ppl(model, bs=1, device=device, block_size=128, model_path=args.model_path))


    if args.zero_shot:
        from lm_eval.tasks import TaskManager
        from lm_eval.utils import make_table
        from lm_eval.models.huggingface import HFLM
        import lm_eval

        task_manager = TaskManager()
        tasks = task_manager.match_tasks(args.tasks)
        hflm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.eval_batchsize,
            max_batch_size="auto",
            trust_remote_code=True,

        )
        results = lm_eval.simple_evaluate(
            hflm, tasks=tasks, batch_size="auto", max_batch_size=256
        )

        metric_vals = {}

        for task, result in results['results'].items():
            task_metrics = {}
            # 处理 acc_norm
            if "acc_norm,none" in result:
                task_metrics["acc_norm,none"] = round(result["acc_norm,none"], 4)
                task_metrics["acc_norm_stderr,none"] = round(
                    result.get("acc_norm_stderr,none", 0.0), 4
                )

            if "acc,none" in result:
                task_metrics["acc,none"] = round(result["acc,none"], 4)
                task_metrics["acc_stderr,none"] = round(
                    result.get("acc_stderr,none", 0.0), 4
                )

            metric_vals[task] = task_metrics

        logger.info("\n" + make_table(results))


if __name__ == "__main__":
    main()
