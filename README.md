# KAIROS Experiments by the Colorado Team

SRL Code adapted from Tao Li's codebase at: [utahnlp/marginal_srl_with_semlink](https://github.com/utahnlp/marginal_srl_with_semlink)

## Contents
1. [Getting Started](#getting-started)
2. [SRL](#srl)
   - [James Gung's Verbnet Parser](#james-gungs-verbnet-parser)
3. [Event Trigger Identification](#event-trigger-identification)
   - [ECB+ T5 Tagger](#ecb-t5-tagger)

## Getting Started

- Dependencies
    ```shell
    pip intall -r requirements.txt
    ```

- Install `cu_kairos` package
    ```shell
    pip install .
    ```

## SRL

### James Gung's Verbnet Parser

This can also be found in [examples/example.py](examples/example.py)
```python
from cu_kairos.srl import jgung_srl

service_url = "http://67.176.72.197:4040/predict/semantics"

sentences = ["I like this sentence and hate this sentence ."]
srl_out = jgung_srl(sentences, url=service_url)
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

You may also run the provided docker on: https://github.com/jgung/verbnet-parser, i.e.

```shell
sudo docker pull jgung/verbnet-parser:0.1-SNAPSHOT
sudo docker run -p 8080:8080 jgung/verbnet-parser:0.1-SNAPSHOT
```

Then the `service_url` needs to be `http://localhost:8080/predict/semantics`

## Event Trigger Identification

### ECB+ T5 Tagger
Using the T5-small model [ahmeshaf/ecb_tagger_seq2seq](https://huggingface.co/ahmeshaf/ecb_tagger_seq2seq) trained on [ahmeshaf/ecb_plus_ed](https://huggingface.co/datasets/ahmeshaf/ecb_plus_ed). Checkout [trainer.py](cu_kairos/evt_tagger/hf/trainer.py) for more info

This achieves 80% F1 in trigger identification on the dev set of ECB+

Example usage found in [examples/ecb_tagger_example.py](examples/ecb_tagger_example.py)

```python
from cu_kairos.evt_tagger import ecb_tagger

sentences = [
    "I like this sentence and hate this sentence and I like this thing",
    "The earthquake took 10 lives .",
]

triggers = ecb_tagger(sentences)

print(triggers)

# [
#   [('like', (2, 6)), ('hate', (25, 29)), ('like', (50, 54))],
#   [('earthquake', (4, 14)), ('took', (15, 19))]
# ]
```