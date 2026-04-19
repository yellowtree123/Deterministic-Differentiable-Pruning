import argparse
import torch
from transformers.pytorch_utils import prune_linear_layer


def get_layers(model):
    return model.model.layers


def keep_from_indicator(indicator: torch.Tensor, min_keep: int = 1):
    """
    indicator: 1/nonzero = keep, 0 = prune
    """
    keep = torch.nonzero(indicator != 0, as_tuple=False).flatten().long()
    if keep.numel() < min_keep:
        k = min(min_keep, indicator.numel())
        keep = torch.topk(indicator.abs().float(), k=k, largest=True).indices.long()
    return torch.sort(keep).values


def expand_head_indices(head_ids: torch.Tensor, head_dim: int, device):
    """
    Convert head ids into flattened channel ids.
    Example: head_ids=[1,3], head_dim=128
      -> [128..255, 384..511]
    """
    if head_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    all_idx = []
    for h in head_ids.tolist():
        all_idx.append(
            torch.arange(
                h * head_dim,
                (h + 1) * head_dim,
                dtype=torch.long,
                device=device,
            )
        )
    return torch.cat(all_idx, dim=0)


def mul_rows_(linear, row_multiplier: torch.Tensor):
    """
    In-place row scaling:
      weight: [out_features, in_features]
      row_multiplier: [out_features]
    """
    row_multiplier = row_multiplier.to(device=linear.weight.device, dtype=linear.weight.dtype)
    assert row_multiplier.numel() == linear.weight.size(0), (
        row_multiplier.shape, linear.weight.shape
    )
    with torch.no_grad():
        linear.weight.mul_(row_multiplier[:, None])
        if linear.bias is not None:
            linear.bias.mul_(row_multiplier)


def prune_one_layer(config, layer, head_indicator, intermediate_indicator):
    attn = layer.self_attn
    mlp = layer.mlp
    device = attn.q_proj.weight.device

    head_dim = config.head_dim

    head_indicator = head_indicator.to(device)
    intermediate_indicator = intermediate_indicator.to(device)

    # attention: structural equivalent of head masking
    keep_q_heads = keep_from_indicator(head_indicator, min_keep=1)
    keep_kv_heads = keep_q_heads

    q_idx = expand_head_indices(keep_q_heads, head_dim, device)
    kv_idx = expand_head_indices(keep_kv_heads, head_dim, device)

    v_row_multiplier = torch.repeat_interleave(head_indicator, head_dim)
    mul_rows_(attn.v_proj, v_row_multiplier)

    attn.q_proj = prune_linear_layer(attn.q_proj, q_idx, dim=0)
    attn.k_proj = prune_linear_layer(attn.k_proj, kv_idx, dim=0)
    attn.v_proj = prune_linear_layer(attn.v_proj, kv_idx, dim=0)
    attn.o_proj = prune_linear_layer(attn.o_proj, q_idx, dim=1)

    attn.num_heads = int(keep_q_heads.numel())
    if hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_heads = int(keep_kv_heads.numel())
    if hasattr(attn, "num_key_value_groups") and hasattr(attn, "num_key_value_heads"):
        attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads

    # mlp: structural equivalent of intermediate masking
    keep_mlp = keep_from_indicator(intermediate_indicator, min_keep=1)

    up_row_multiplier = intermediate_indicator.to(device=device, dtype=mlp.up_proj.weight.dtype)
    mul_rows_(mlp.up_proj, up_row_multiplier)

    mlp.gate_proj = prune_linear_layer(mlp.gate_proj, keep_mlp, dim=0)
    mlp.up_proj   = prune_linear_layer(mlp.up_proj,   keep_mlp, dim=0)
    mlp.down_proj = prune_linear_layer(mlp.down_proj, keep_mlp, dim=1)

    return {
        "q_heads": int(keep_q_heads.numel()),
        "kv_heads": int(keep_kv_heads.numel()),
        "intermediate": int(keep_mlp.numel()),
    }


def actual_prune(hf_dense_model, head_z, intermediate_z):
    layers = get_layers(hf_dense_model)

    assert head_z.size(0) == len(layers), (head_z.size(), len(layers))
    assert intermediate_z.size(0) == len(layers), (intermediate_z.size(), len(layers))

    for i, layer in enumerate(layers):
        stat = prune_one_layer(
            hf_dense_model.config,
            layer,
            head_indicator=head_z[i],
            intermediate_indicator=intermediate_z[i],
        )
        print(f"layer {i:02d}: {stat}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dict_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--head_key",
        type=str,
        default="module.l0_module.masks.head.z_loga",
    )
    parser.add_argument(
        "--intermediate_key",
        type=str,
        default="module.l0_module.masks.intermediate.z_loga",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--safe_serialization", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    sd = torch.load(args.state_dict_path, map_location="cpu")

    if args.head_key not in sd:
        raise KeyError(f"Missing head key: {args.head_key}")
    if args.intermediate_key not in sd:
        raise KeyError(f"Missing intermediate key: {args.intermediate_key}")

    head_z = torch.relu(sd[args.head_key])
    intermediate_z = torch.relu(sd[args.intermediate_key])

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )

    actual_prune(model, head_z, intermediate_z)
    model.save_pretrained(
        args.output_dir,
        safe_serialization=args.safe_serialization,
    )