import json
import pandas as pd

from pathlib import Path


def ensure_directory(directory: Path):
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True)


def to_csv(dict_array, file_path):
    pd.DataFrame(dict_array).to_csv(file_path, index=False)


def read_json_lines(jsonl_file):
    with open(jsonl_file) as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]
