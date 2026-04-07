import argparse
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer


def load_raw_datasets(args):
    if args.dataset_name is not None:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        dataset_args = {}

        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = "json" if args.train_file.endswith(".json.gz") else args.train_file.split(".")[-1]

        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]

        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=args.cache_dir,
            **dataset_args,
        )

    if "validation" not in raw_datasets:
        if args.dataset_name is not None:
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir,
            )
        else:
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
                **dataset_args,
            )

    return raw_datasets


def main(args):
    raw_datasets = load_raw_datasets(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=not args.use_slow_tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    if args.use_alpaca_format:
        def alpaca_format(example):
            inst = (example.get("instruction") or "").strip()
            resp = (example.get("response") or "").strip()
            eos = tokenizer.eos_token or ""
            return {
                "text": f"### Instruction:\n{inst}\n\n### Response:\n{resp}{eos}"
            }

        raw_datasets = raw_datasets.map(
            alpaca_format,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Formatting to Alpaca style",
        )
        text_column_name = "text"
    else:
        train_columns = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in train_columns else train_columns[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    block_size = args.block_size or tokenizer.model_max_length
    block_size = min(block_size, tokenizer.model_max_length)

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    lm_datasets.save_to_disk(args.output_dir)
    print("Saved tokenized dataset to:", args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--validation_file", type=str, default=None)
    parser.add_argument("--validation_split_percentage", type=int, default=5)

    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=8)
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--use_slow_tokenizer", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_alpaca_format", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/cache")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--no_keep_linebreaks", action="store_true")

    args = parser.parse_args()
    main(args)