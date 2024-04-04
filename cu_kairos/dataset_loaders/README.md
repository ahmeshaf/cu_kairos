# A Unified Events Seq2Seq HuggingFace Dataset Creator

Checkout https://docs.google.com/spreadsheets/d/1RtsYdDgDSFwBANcvFNM4w-_XwbwG2Im4RcnXkfKSoxU/edit#gid=0 for dataset info

## Trigger Identification
TODO: Take any dataset and create of the form [ahmeshaf/ecb_plus_ed](https://huggingface.co/datasets/ahmeshaf/ecb_plus_ed)

### Freely Avaialable Datasets
1. ECB+
```shell
python -m free.ecb create-ecb-tagging-dataset \
          mention_map_path \
          output_folder
```