import pandas as pd

from pathlib import Path


def ensure_directory(directory: Path):
    if not Path(directory).exists():
        Path(directory).mkdir(parents=True)


def to_csv(dict_array, file_path):
    pd.DataFrame(dict_array).to_csv(file_path, index=False)
