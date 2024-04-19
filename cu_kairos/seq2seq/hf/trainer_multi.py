# Description: Custom Trainer class for multi eval model training

import json
import numpy as np
import torch
import typer
from datasets import concatenate_datasets, load_dataset
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.utils import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction
    from transformers.training_args import TrainingArguments

from .trainer import pre_process_eos

logger = logging.get_logger(__name__)


class MultiEvalTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        args: "TrainingArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_datasets: Optional[Dict[str, Union[Dataset, Dict[str, Dataset]]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        self.eval_datasets = eval_datasets
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **gen_kwargs,
    ) -> Union[Dict[str, float], Dict]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        eval_scores = {}
        for dataset_name, eval_dataset_ in self.eval_datasets.items():
            print(f"Evaluating on {dataset_name}")
            eval_scores[dataset_name] = super(MultiEvalTrainer, self).evaluate(
                eval_dataset=eval_dataset_,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        return eval_scores


app = typer.Typer()


@app.command()
def trainer_seq2seq_multi(config_file: Path, datasets_dict: Dict[str, Dataset]):
    config = json.load(open(config_file))
    # print(dataset_names)
    # datasets_dict = {d_name: load_dataset(d_name) for d_name in dataset_names}

    model_name_or_path = config.pop("model_name_or_path")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Concatenate the train sets of each dataset

    train_dataset = concatenate_datasets(
        [
            pre_process_eos(dataset["train"], tokenizer.eos_token)
            for dataset in datasets_dict.values()
        ]
    )

    eval_datasets = {
        dataset_name: pre_process_eos(dataset["dev"], tokenizer.eos_token)
        for dataset_name, dataset in datasets_dict.items()
    }

    def preprocess_data(examples):
        model_inputs = tokenizer(examples["prompt"], max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["response"], max_length=32, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tokenized = train_dataset.map(preprocess_data, batched=True)
    evals_tokenized = {
        dataset_name: eval_dataset.map(preprocess_data, batched=True)
        for dataset_name, eval_dataset in eval_datasets.items()
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")
    # from peft import prepare_model_for_kbit_training
    # from peft import LoraConfig, get_peft_model, TaskType
    #
    # model = prepare_model_for_kbit_training(model)
    #
    # lora_config = LoraConfig(
    #     r=16, lora_alpha=32, target_modules=["q", "v"], lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
    # )
    #
    # model = get_peft_model(model, lora_config)

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

    t5_trainer = MultiEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_datasets=evals_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_prf,
    )
    t5_trainer.train()
    t5_trainer.save_model()


if __name__ == "__main__":
    app()
