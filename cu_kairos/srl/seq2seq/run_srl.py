import regex

from datasets import Dataset
from difflib import SequenceMatcher
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import pipeline, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, GenerationConfig
from transformers.pipelines.pt_utils import KeyDataset
from typing import List


# ----------- Helper Functions ----------- #
def find_best_match(sentence, phrase, claimed_ranges):
    # Adjust the regex to consider word boundaries. The \b ensures that we match whole words
    # but only where it makes sense based on the phrase itself.
    pattern = (
        r"\b%s\b" % regex.escape(phrase)
        if phrase[0].isalnum() and phrase[-1].isalnum()
        else r"%s" % regex.escape(phrase)
    )
    matches = regex.finditer(f"({pattern}){{e<=3}}", sentence, overlapped=True)
    best_match = None
    highest_ratio = 0.0

    for match in matches:
        start, end = match.span()
        # Exclude matches that overlap with claimed ranges
        if not any(
            start < cr_end and end > cr_start for cr_start, cr_end in claimed_ranges
        ):
            match_ratio = SequenceMatcher(None, match.group(), phrase).ratio()
            if match_ratio > highest_ratio:
                highest_ratio = match_ratio
                best_match = match

    return best_match


# noinspection DuplicatedCode
def find_phrase_offsets_fuzzy(sentence, phrases):
    results = []
    claimed_ranges = []
    for phrase in phrases:
        match = find_best_match(sentence, phrase, claimed_ranges)
        if match:
            start, end = match.span()
            # Claim this range
            claimed_ranges.append((start, end))
            results.append((match.group(), start, end))
    return results


def get_model_tokenizer_generation_config(model_name, is_peft=False):
    if is_peft:
        config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_name)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    generation_config = GenerationConfig.from_pretrained(model_name)
    return model, tokenizer, generation_config


def get_prf(gold_tags: list, predicted_tags: list):
    # convert to sets
    gold_tags_set = set(gold_tags)
    predicted_tags_set = set(predicted_tags)
    # calculate true positives
    true_positives = len(gold_tags_set.intersection(predicted_tags_set))
    # calculate precision
    precision = true_positives / len(predicted_tags_set) if len(predicted_tags_set) > 0 else 0
    # calculate recall
    recall = true_positives / len(gold_tags_set) if len(gold_tags_set) > 0 else 0
    # calculate f1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


# ----------- Main Functions ----------- #

def tag_with_prompts(tagger_model_name, prompts, batch_size=32, is_peft=False):
    tagger_dataset = Dataset.from_dict({"prompt": prompts})
    model, tokenizer, generation_config = get_model_tokenizer_generation_config(
        tagger_model_name, is_peft
    )
    tagger_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        generation_config=generation_config,
        device_map="auto",
    )
    tagger_out = []
    for out in tqdm(
        tagger_pipe(KeyDataset(tagger_dataset, "prompt"), batch_size=batch_size),
        total=len(prompts),
        desc="Tagging",
    ):
        tagger_out.append(out[0]["generated_text"])
    return tagger_out


def tagger(
    tagger_model_name: str,
    sentences: List[str],
    men_type: str = "evt",
    batch_size: int = 32,
    is_peft: bool = False,
):
    if men_type == "ent":
        prompts = [f"entities: {sentence}" for sentence in sentences]
    else:
        prompts = [f"triggers: {sentence}" for sentence in sentences]

    tagger_out = tag_with_prompts(tagger_model_name, prompts, batch_size, is_peft)

    predicted_mentions = []

    for i, (sentence, tags_str) in enumerate(zip(sentences, tagger_out)):
        split_tags = [t.strip() for t in tags_str.split("|") if t.strip() != ""]
        offsets = find_phrase_offsets_fuzzy(sentence, split_tags)
        predicted_mentions.append(list(zip(split_tags, offsets)))

    return predicted_mentions


def generate_srl_prompts(sentences, sentences_triggers):
    for i, (sentence, triggers) in enumerate(zip(sentences, sentences_triggers)):
        for trigger in triggers:
            (_, (trigger_txt, start, end)) = trigger
            sentence_prompt = sentence[:start] + f"[{trigger_txt}]" + sentence[end:]
            srl_prompt = f"SRL for [{trigger[0]}]: {sentence_prompt}"
            yield i, trigger, srl_prompt, sentence


def srl_predicted_triggers(
    sentences, triggers, srl_model, batch_size, is_srl_peft=False
):
    srl_prompts = list(generate_srl_prompts(sentences, triggers))
    s_ids, flattened_triggers, srl_prompts, flattened_sentences = zip(*srl_prompts)
    srl_dataset = Dataset.from_dict({"prompt": srl_prompts})

    model_s, tokenizer_s, generation_config_s = get_model_tokenizer_generation_config(
        srl_model, is_srl_peft
    )

    srl_pipe = pipeline(
        "text2text-generation",
        model=model_s,
        tokenizer=tokenizer_s,
        generation_config=generation_config_s,
    )
    outputs = []
    for out in tqdm(
            srl_pipe(KeyDataset(srl_dataset, "prompt"), batch_size=batch_size),
            total=len(srl_dataset),
            desc="SRL",
    ):
        outputs.append(out[0]["generated_text"])

    s_id2srl = {}

    for i, trigger, output, sentence in zip(
        s_ids, flattened_triggers, outputs, flattened_sentences
    ):
        if i not in s_id2srl:
            s_id2srl[i] = []

        s_id2srl[i].append((trigger, sentence, output))

    s_id2srl = sorted(s_id2srl.items(), key=lambda x: x[0])
    sentence_srls = [srl for _, srl in s_id2srl]

    triggers_srls_offsets = []

    for sentence_srl in sentence_srls:
        curr_triggers_srls_phrases = []
        for trigger, sentence, str_srl in sentence_srl:
            arg_srls = [tuple(arg.split(": ")) for arg in str_srl.split(" | ")]
            arg_labels, arg_phrases = zip(*arg_srls)
            phrase_offsets = find_phrase_offsets_fuzzy(sentence, arg_phrases)
            arg_srls = list(zip(arg_labels, arg_phrases, phrase_offsets))
            curr_triggers_srls_phrases.append((trigger, arg_srls))
        triggers_srls_offsets.append(curr_triggers_srls_phrases)

    return triggers_srls_offsets


def semantic_role_labeler_seq2seq(
    sentences: List[str],
    trigger_model_name: str = "ahmeshaf/ecb_tagger_seq2seq",
    is_trigger_peft: bool = False,
    srl_model: str = "cu-kairos/propbank_srl_seq2seq_t5_large",
    is_srl_peft: bool = False,
    batch_size: int = 32,
):
    triggers = tagger(
        trigger_model_name,
        sentences,
        batch_size=batch_size,
        men_type="evt",
        is_peft=is_trigger_peft,
    )
    return srl_predicted_triggers(
        sentences, triggers, srl_model, batch_size, is_srl_peft=is_srl_peft
    )


if __name__ == "__main__":
    sentences_ = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    print(semantic_role_labeler_seq2seq(sentences_))
