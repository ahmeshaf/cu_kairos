import json
import pandas as pd
import pickle
import typer

from pathlib import Path


app = typer.Typer()


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

    split2datapoints = {}
    for sent_id, mids in sentence2mids.items():
        sentence, split = sent_id2sentence[sent_id]
        if split not in split2datapoints:
            split2datapoints[split] = []
        triggers = [
            mention_map[m_id]["mention_text"]
            for m_id in mids
            if mention_map[m_id]["men_type"] == "evt"
        ]
        entities = [
            mention_map[m_id]["mention_text"]
            for m_id in mids
            if mention_map[m_id]["men_type"] == "ent"
        ]
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


@app.command()
def create_mention_dataset(mention_map_path: Path, output_folder: Path):
    mention_map = pickle.load(open(mention_map_path, "rb"))

    if not output_folder.exists():
        output_folder.mkdir(parents=True)

    for mention in mention_map.values():
        if mention["split"] == "debug_split":
            mention["split"] = "dev_small"

    splits = ["train", "dev", "dev_small", "test"]

    useful_keys = [
        "mention_id",
        "split",
        "men_type",
        "doc_id",
        "sentence_id",
        "sentence",
        "start_char",
        "end_char",
        "mention_text",
        "gold_cluster",
        "lemma",
        "sentence_tokens",
        "marked_sentence",
        "marked_doc",
    ]

    for split in splits:
        split_m_ids = [
            m_id for m_id, mention in mention_map.items() if mention["split"] == split
        ]
        split_mentions = [
            {key: mention_map[m_id][key] for key in useful_keys} for m_id in split_m_ids
        ]
        json.dump(split_mentions, open(output_folder / f"{split}.json", "w"), indent=1)


if __name__ == "__main__":
    app()
