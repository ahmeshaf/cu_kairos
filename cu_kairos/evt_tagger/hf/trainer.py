import json
import torch
import typer

from datasets import load_from_disk
from pathlib import Path
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from .models import EventTagger
from ..util import compute_metrics

app = typer.Typer()


class EventTaggingDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, label_to_id, max_length=512):
        self.sentences = sentences  # List of sentences
        self.labels = labels  # List of label sequences in IOB format
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label_sequence = self.labels[idx]
        inputs = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = [self.label_to_id["O"]] * len(
            inputs["input_ids"]
        )  # Initialize with "O"

        input_ids = inputs["input_ids"]
        offset_mapping = inputs["offset_mapping"]
        token_index = 0

        for i, (start, end) in enumerate(offset_mapping):
            if start == end == 0 or token_index >= len(label_sequence):
                continue  # Skip special tokens and handle index overflow

            # Align labels with first subtoken of original token
            if input_ids[i] != self.tokenizer.pad_token_id:
                labels[i] = self.label_to_id.get(
                    label_sequence[token_index], self.label_to_id["O"]
                )
                if (
                    token_index < len(label_sequence) - 1
                    and "B-" in label_sequence[token_index]
                ):
                    next_label = label_sequence[token_index + 1]
                    if "I-" in next_label:
                        token_index += 1
                        while (
                            token_index < len(label_sequence)
                            and "I-" in label_sequence[token_index]
                        ):
                            token_index += 1
                            continue
                else:
                    token_index += 1

        inputs["labels"] = labels
        inputs.pop("offset_mapping")  # No longer needed
        return {key: torch.tensor(val) for key, val in inputs.items()}


@app.command()
def trainer(config_file: Path, train_dataset_path: str, eval_dataset_path: str):
    """
    Train the event tagger

    :param config_file: path to the file containing training configs
    :param train_dataset_path: path to the train_dataset path
    :param eval_dataset_path: path to the eval_dataset_path
    :return:
    """
    config = json.load(open(config_file))
    # load the datasets
    train_dataset = load_from_disk(train_dataset_path)
    eval_dataset = load_from_disk(eval_dataset_path)
    model_name_or_path = config.pop("model_name_or_path")
    labels = config.pop("labels")
    training_args = TrainingArguments(**config)

    model = EventTagger.from_pretrained(model_name_or_path, num_labels=len(labels), config=config)

    evt_trainer = Trainer(
        model=model,  # Your initialized RoBERTa model
        args=training_args,  # Training arguments defined above
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=eval_dataset,  # Evaluation dataset
        compute_metrics=compute_metrics,   # Function to compute metrics, for evaluation
    )
    evt_trainer.train()


if __name__ == "__main__":
    app()
