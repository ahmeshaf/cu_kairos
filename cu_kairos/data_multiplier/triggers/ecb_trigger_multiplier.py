import json
import random

import datasets
import pickle
import typer

from langchain.chains.llm import LLMChain
from langchain.output_parsers import StructuredOutputParser
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from ..util import JsonParser, find_word_offsets
from ...util import ensure_directory

app = typer.Typer()

# synonyms generation
syn_prompt = '''
You are a language expert. Your task is to replace specific words in a given sentence to their synonyms. Here are the detailed steps you need to follow:

Read the Sentence Provided: Focus on understanding the context and meaning of the sentence.
Review the Word List: This list contains the words you need to transform.
Generate 5 distinct Synonyms for each word in the list : 
Compose a New Sentence:
Replace the original words with your chosen metaphors randomly.
Ensure the new sentence maintains logical and grammatical coherence.
Sentence to Transform:
"""{sentence}"""

Word List to Convert into Synonyms:
"""{triggers}"""

Output Requirements:
Provide your final output in JSON format, including:

The "Original Sentence".
The "Original Word List".
The "Synonymous Word List" (a dictionary of Original Word to the Synonymous Words).
The "Paraphrased Sentence" (the sentence with synonymous word incorporated).
Remember, the goal is to use synonyms to convey the original sentence's meaning without altering the core information.
'''
synonymous_prompt = PromptTemplate(
    template=syn_prompt,
    input_variables=["sentence", "triggers"],
)


def structured_output() -> StructuredOutputParser:
    return JsonParser()


@app.command()
def save_synonyms_gpt(
    output_file: Path,
    split: str,
    model_name: str = "gpt-4-0125-preview",
    temperature: float = 0.75,
    cache_folder: str = "/tmp/ecb_syns/",
    experiment_name: str = "synonyms_base",
    force: bool = False,
):
    cache_folder = Path(cache_folder)
    ensure_directory(cache_folder)
    ensure_directory(output_file.parent)
    cache_file = cache_folder / f"ecb_{split}_{model_name}_{experiment_name}.pkl"

    if cache_file.exists() and not force:
        raw_cache = pickle.load(open(cache_file, "rb"))
    else:
        raw_cache = {}

    dataset = datasets.load_dataset("ahmeshaf/ecb_plus_ed")
    dataset_split = dataset[split]
    dataset_size = len(dataset_split)

    parser = JsonOutputParser()

    # Initialize the result dict
    result_dict = {}

    # Initialize the LLM
    llm = ChatOpenAI(temperature=temperature, model=model_name, request_timeout=180)
    chain = LLMChain(llm=llm, prompt=synonymous_prompt)

    for data in tqdm(dataset_split, desc="Running Syns", total=dataset_size):
        # print(data["sentence_id"], data["sentence"], data["triggers"])
        formatted_prompt = synonymous_prompt.format_prompt(
            sentence=data["sentence"], triggers=data["triggers"]
        )
        sentence_id = data["sentence_id"]

        if sentence_id == "sentence_id":
            continue

        if sentence_id in raw_cache:
            predict = raw_cache[sentence_id]["predict"]
        else:
            with get_openai_callback() as cb:
                predict = chain.run(
                    sentence=data["sentence"], triggers=data["triggers"]
                )

                predict_cost = {
                    "Total": cb.total_tokens,
                    "Prompt": cb.prompt_tokens,
                    "Completion": cb.completion_tokens,
                    "Cost": cb.total_cost,
                }
                raw_cache[sentence_id] = {
                    "predict": predict,
                    "predict_cost": predict_cost,
                    "format_prompt": formatted_prompt,
                }
                pickle.dump(raw_cache, open(cache_file, "wb"))

        try:
            predict_dict = parser.parse(predict)
            predict_dict["format_prompt"] = formatted_prompt
            result_dict[sentence_id] = predict_dict
            # print(predict_dict)
        except Exception as e:
            print(sentence_id, e)

    pickle.dump(result_dict, open(output_file, "wb"))


@app.command()
def generate_sentences_deterministic(
    result_dict_path: Path, output_json_path: Path, n_iters: int = 10
):
    ensure_directory(output_json_path.parent)

    result_dict = pickle.load(open(result_dict_path, "rb"))
    output_dict = []
    for _ in tqdm(list(range(n_iters)), desc="multiplying", total=n_iters):
        for sentence_id, result in result_dict.items():
            sentence = result["Original Sentence"]
            triggers = result["Original Word List"]
            synonyms_dict = result["Synonymous Word List"]
            # paraphrased_sentence = result["Paraphrased Sentence"]

            replacement_map = []

            # Find matching synonyms in the Paraphrased Sentence
            for original_word, synonyms in list(synonyms_dict.items()):
                if original_word.lower() in ["this", "that", "it", "which"]:
                    synonyms_dict[original_word] = [original_word]
                # Sample a random synonym
                replacement_map.append(
                    (original_word, random.sample(synonyms, 1)[0])
                )

            original_offsets = find_word_offsets(sentence, triggers)
            w_t_offsets = list(zip(replacement_map, original_offsets))
            w_t_offsets = [(w_t, offset) for w_t, offset in w_t_offsets if offset]
            new_sentence = ""
            end_index = 0
            for (original_word, new_word), offset in w_t_offsets:
                if offset:
                    new_sentence += sentence[end_index:offset[0]] + new_word
                    end_index = offset[1]
            new_sentence += sentence[end_index:]
            old_sentence_dict = {
                "sentence": "triggers: " + sentence,
                "labels": " | ".join(triggers)
            }
            new_sentence_dict = {
                "sentence": "triggers: " + new_sentence,
                "labels": " | ".join([map_[-1] for map_ in replacement_map]),
            }
            output_dict.append(old_sentence_dict)
            output_dict.append(new_sentence_dict)
            # print(old_sentence_dict)
    print(len(output_dict))
    json.dump(output_dict, open(output_json_path, "w"), indent=1)


if __name__ == "__main__":
    app()
