import json

import datasets
import numpy as np
import torch
import typer

from datasets import load_from_disk, Dataset
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from .models import EventTagger
from ..util import compute_metrics

app = typer.Typer()


def pre_process(dataset, eos_token):
    entity_prefix = "entities: "
    triggers_prefix = "triggers: "
    inputs = [entity_prefix + doc for doc in dataset["sentence"]] + [
        triggers_prefix + doc for doc in dataset["sentence"]
    ]
    labels_txt = [doc if doc else "NA" for doc in dataset["entities"]] + [
        doc if doc else "NA" for doc in dataset["triggers"]
    ]

    labels_txt = [doc + " " + eos_token for doc in labels_txt]

    return Dataset.from_dict({"sentence": inputs, "labels": labels_txt})


def pre_process_triggers(dataset, eos_token):
    triggers_prefix = "triggers: "
    inputs = [triggers_prefix + doc for doc in dataset["sentence"]]
    labels_txt = [doc if doc else "NA" for doc in dataset["triggers"]]

    labels_txt = [doc + " " + eos_token for doc in labels_txt]

    return Dataset.from_dict({"sentence": inputs, "labels": labels_txt})


@app.command()
def trainer_seq2seq(config_file: Path, dataset_name: str):
    config = json.load(open(config_file))
    dataset = datasets.load_dataset(dataset_name)

    model_name_or_path = config.pop("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    train_dataset = pre_process(dataset["train"], tokenizer.eos_token)
    eval_dataset = pre_process_triggers(dataset["validation"], tokenizer.eos_token)

    def preprocess_data(examples):
        model_inputs = tokenizer(examples["sentence"], max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["labels"], max_length=32, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_data, batched=True)
    eval_tokenized = eval_dataset.map(preprocess_data, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    training_args = Seq2SeqTrainingArguments(**config["trainer"])

    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    generation_config.eos_token_id = tokenizer.eos_token_id
    generation_config.update(**config["generation"])

    training_args.generation_config = generation_config
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    def compute_prf(eval_pred):
        predictions, labs = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labs = np.where(labs != -100, labs, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labs, skip_special_tokens=True)
        # print(decoded_labels)

        decoded_preds = [
            set([p.strip() for p in pred.split("|") if p.strip() != ""])
            for pred in decoded_preds
        ]
        decoded_labels = [
            set([p.strip() for p in pred.split("|") if p.strip() != ""])
            for pred in decoded_labels
        ]

        common_preds_lens = [
            len(set.intersection(p1, p2))
            for p1, p2 in zip(decoded_preds, decoded_labels)
        ]
        decoded_preds_lens = [len(p) for p in decoded_preds]
        decoded_labels_lens = [len(p) for p in decoded_labels]

        return {
            "precision": np.sum(common_preds_lens) / np.sum(decoded_preds_lens),
            "recall": np.sum(common_preds_lens) / np.sum(decoded_labels_lens),
        }

    t5_trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_prf,
    )
    t5_trainer.train()
    t5_trainer.save_model()


@app.command()
def train_ecb_ed(config_file: Path):
    pass


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

    model = EventTagger.from_pretrained(
        model_name_or_path, num_labels=len(labels), config=config
    )

    evt_trainer = Trainer(
        model=model,  # Your initialized RoBERTa model
        args=training_args,  # Training arguments defined above
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=eval_dataset,  # Evaluation dataset
        compute_metrics=compute_metrics,  # Function to compute metrics, for evaluation
    )
    evt_trainer.train()


if __name__ == "__main__":
    app()
