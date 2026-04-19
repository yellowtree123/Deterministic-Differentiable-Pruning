python actual_prune/prune_and_save.py \
  --state_dict_path /root/autodl-tmp/runs/llama-7b-hf_20260419_164032/testrun/step_1000/trainable_params.pt \
  --model_name_or_path llama-7b-hf \
  --output_dir /root/autodl-tmp/llama-7b-hf-pruned \
  --trust_remote_code \
  --safe_serialization
