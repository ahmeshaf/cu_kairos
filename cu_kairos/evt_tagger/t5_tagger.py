import numpy as np

from transformers import (
    GenerationConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Pipeline,
)
from transformers.pipelines.base import PipelineException


class TriggersPipeline(Pipeline):
    def __init__(self, model, tokenizer, **kwargs):
        self.preprocess_params, self.forward_params, self.postprocess_params = (
            self._sanitize_parameters(**kwargs)
        )
        super().__init__(model, tokenizer, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        preprocess_params = {}
        postprocess_params = {}
        forward_params = {}

        for param_name, param_value in pipeline_parameters.items():
            if "max_length" == param_name:
                preprocess_params[param_name] = param_value
            if "generation_config" == param_name:
                forward_params[param_name] = param_value

        return preprocess_params, forward_params, postprocess_params

    def preprocess(self, prompt):
        # Preprocessing: Add the 'triggers: ' prefix to each prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        if not isinstance(prompt, list):
            raise PipelineException(
                "The `prompt` argument needs to be of type `str` or `list`."
            )
        prefixed_prompt = ["triggers: " + p for p in prompt]
        return self.tokenizer(
            prefixed_prompt,
            truncation=True,
            return_tensors="pt",
            **self.preprocess_params
        )

    def _forward(self, model_inputs, **forward_params):
        # This step is necessary if you need to adjust how the model is called, for example, to modify the forward pass
        return self.model.generate(**model_inputs, **forward_params)

    def postprocess(self, predictions, **postprocess_params):
        # Postprocess the model output if needed
        predictions = np.where(
            predictions != -100, predictions, self.tokenizer.pad_token_id
        )
        decoded_preds = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True
        )
        decoded_pred = [s.strip() for s in decoded_preds[0].split("|") if s.strip() != ""]
        return decoded_pred

    def __call__(self, inputs, **kwargs):
        # Ensure inputs is a list for consistent preprocessing
        return super().__call__(inputs, **kwargs)


def find_word_offsets(sentence, words):
    # Initialize a list to store the results
    offsets = []

    # Initialize the start search position
    search_pos = 0

    # Loop through each word to find its position in the sentence
    for word in words:
        # Find the position of the word in the sentence starting from search_pos
        word_pos = sentence.find(word, search_pos)

        # If the word is found, append its start and end positions to the offsets list
        if word_pos != -1:
            offsets.append((word_pos, word_pos + len(word)))

            # Update the search position to just after the current word's position
            search_pos = word_pos + len(word)
        else:
            # If word is not found, append None or an indicator
            offsets.append(None)

    return offsets


def ecb_tagger(sentences, model_name="ahmeshaf/ecb_tagger_seq2seq"):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    # Initialize our custom pipeline
    triggers_pipeline = TriggersPipeline(
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        generation_config=generation_config,
    )
    sentence_triggers = triggers_pipeline(sentences)
    trigger_offsets = []
    for sentence, triggers in zip(sentences, sentence_triggers):
        offsets = find_word_offsets(sentence, triggers)
        trigger_offsets.append(list(zip(triggers, offsets)))
    return trigger_offsets


if __name__ == "__main__":
    print(
        ecb_tagger(
            [
                "I like this sentence and hate this sentence and I like this thing",
                "The earthquake took 10 lives ."
            ]
        )
    )
