# KAIROS Experiments by the Colorado Team

SRL Code adapted from Tao Li's codebase at: [utahnlp/marginal_srl_with_semlink](https://github.com/utahnlp/marginal_srl_with_semlink)

## Getting Started

- Dependencies
    ```shell
    pip intall -r requirements.txt
    ```

- Install `cu_kairos` package
    ```shell
    pip install .
    ```

## Usage

This can also be found in [examples/example.py](examples/example.py)
```python
from cu_kairos.srl.tao_li import run_srl

sentences = ["I like this example sentence ."]
srl_out = run_srl(sentences)
print(srl_out)
# [(['like', 'like.01', 2, 6], [('ARG0', 'I', 0, 1), ('ARG1', 'this example sentence', 7, 27)])]
```