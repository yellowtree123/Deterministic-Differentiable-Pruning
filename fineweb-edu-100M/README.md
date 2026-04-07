---
dataset_info:
  features:
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 552296826
    num_examples: 115482
  download_size: 328715464
  dataset_size: 552296826
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
---


## Sampling Methodology

This dataset was created using **reservoir sampling**, a statistically unbiased random sampling algorithm that guarantees each sample from the source dataset has an equal probability of being included. This ensures the 100M token sample is representative of the full dataset's characteristics.

**Source Dataset**: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
**Sample Size**: 100M tokens
**Content**: Curated educational web resources

Reservoir sampling enables rapid experimentation and ablation studies without processing the entire source dataset, while maintaining statistical validity of results.

For details on how this dataset was used in optimal pre-training data composition research, see the [blog post](https://huggingface.co/blog/codelion/optimal-dataset-mixing/).

## Citation

If you use this model/dataset, please cite:

```bibtex
@article{sharma2025billion,
  title={The 1 Billion Token Challenge: Finding the Perfect Pre-training Mix},
  author={Sharma, Asankhaya},
  year={2025},
  url={https://huggingface.co/blog/codelion/optimal-dataset-mixing/}
}
```

For more details, see the [blog post](https://huggingface.co/blog/codelion/optimal-dataset-mixing/).