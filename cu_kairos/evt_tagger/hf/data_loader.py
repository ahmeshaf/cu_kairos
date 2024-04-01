import pandas as pd
import pickle
import torch
import typer

from pathlib import Path
from torch.utils.data import Dataset


app = typer.Typer()


class EventTaggingDatasetED(Dataset):
    # Tagging dataset in the format of encoder decoder
    def __init__(self, sentences, triggers, tokenizer, max_length=256):
        self.sentences = sentences  # List of sentences
        self.triggers = triggers  # List of label sequences in IOB format
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        triggers = self.triggers[idx]
        inputs = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        triggers_text = " | ".join(triggers)

        labels = self.tokenizer(
            triggers_text,
            padding=True,
            truncation=True,
            max_length=25,
        )

        inputs["labels"] = labels["input_ids"]
        return {key: torch.tensor(val) for key, val in inputs.items()}


class EventTaggingDataset(Dataset):
    # Tagging dataset for IOB Tagging
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
def create_ecb_tagging_dataset(mention_map_path: Path, output_folder: Path):
    mention_map = pickle.load(open(mention_map_path, "rb"))
    sentence2mids = {}
    sent_id2sentence = {}
    for m_id, mention in mention_map.items():
        sentence_id = (mention["doc_id"], str(mention["sentence_id"]))
        if sentence_id not in sentence2mids:
            sentence2mids[sentence_id] = []
        sentence2mids[sentence_id].append(m_id)
        sentence = mention["sentence"]
        split = mention["split"]
        sent_id2sentence[sentence_id] = (sentence, split)

    sentence2mids = {
        sent_id: sorted(
            mids,
            key=lambda x: mention_map[x]["start_char"],
        )
        for sent_id, mids in sentence2mids.items()
    }

    datapoints = []
    split2datapoints = {}
    for sent_id, mids in sentence2mids.items():
        sentence, split = sent_id2sentence[sent_id]
        if split not in split2datapoints:
            split2datapoints[split] = []
        triggers = [mention_map[m_id]["mention_text"] for m_id in mids if mention_map[m_id]["men_type"] == "evt"]
        entities = [mention_map[m_id]["mention_text"] for m_id in mids if mention_map[m_id]["men_type"] == "ent"]
        triggers = " | ".join(triggers)
        entities = " | ".join(entities)
        if entities.strip() == "":
            entities = "<unk>"

        if triggers.strip() == "":
            triggers = "<unk>"

        datapoint = {
            "sentence_id": "_".join(sent_id),
            "sentence": sentence,
            "entities": entities,
            "triggers": triggers,
        }
        split2datapoints[split].append(datapoint)

    for split, data in split2datapoints.items():
        pd.DataFrame(data).to_csv(output_folder / f"{split}.csv", index=False)


if __name__ == "__main__":
    app()
