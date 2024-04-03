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
from cu_kairos.srl import jgung_srl

sentences = ["I like this sentence and hate this sentence ."]
srl_out = jgung_srl(sentences)
print(srl_out)
# [
#   [   
#       (
#           ['like', 'admire-31.2-1', 2, 6], 
#           [('A0', 'I', 0, 1), ('A1', 'this sentence', 7, 20)]
#       ), 
#       (
#           ['hate', 'admire-31.2-1', 25, 29], 
#           [('A0', 'I', 0, 1), ('A1', 'this sentence', 30, 43)]
#       )
#   ]
# ]
```