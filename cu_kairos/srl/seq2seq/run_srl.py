from transformers import pipeline

from ...util import *


def generate_srl_prompts(sentences, sentences_triggers):
    for i, (sentence, triggers) in enumerate(zip(sentences, sentences_triggers)):
        for trigger in triggers:
            sentence = (
                sentence[: trigger[1][0]]
                + f"[{trigger[0]}]"
                + sentence[trigger[1][1]:]
            )
            srl_prompt = f"SRL for [{trigger[0]}]: {sentence}"
            yield i, trigger, srl_prompt


def srler(tagger_model, srl_model, sentences, batch_size=32):
    # get trigger tags
    tagger_pipe = pipeline("text2textgeneration", tagger_model)

    generated_triggers = tagger_pipe(sentences, batch_size=batch_size)

    # Todo: parse the triggers extracting. use the "generated_text" key
    trigger_offsets = parse_triggers(sentences, generated_triggers)

    srl_pipe = pipeline("text2textgeneration", srl_model)

    # Generate srl_prompts from the triggers and sentences
    srl_prompts = generate_srl_prompts(sentences, trigger_offsets)
    s_ids, flattened_triggers, srl_prompts = zip(*srl_prompts)

    generated_srls = srl_pipe(srl_prompts)

    # Todo: parse the srls and return offsets
    # TODO: use find_phrase_offsets_fuzzy from ...util





