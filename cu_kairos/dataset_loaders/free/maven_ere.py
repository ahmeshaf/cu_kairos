import pandas as pd
import typer

from pathlib import Path

from ...util import ensure_directory, read_json_lines


app = typer.Typer()


@app.command()
def create_maven_ere_ed(maven_ere_folder: str, output_folder: Path):
    """
    This function reads the Maven ERE dataset and creates a dataset for the Event Trigger Detection task.
    :param maven_ere_folder: unzipped https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1
    :param output_folder: output folder to save the dataset
    :return:
    """
    ensure_directory(output_folder)
    splits = ["train", "valid", "test"]

    # Load the dataset from the dataset_folder + split.jsonl path  using read_json_lines
    sentence_triggers = []
    for split in splits:
        split_path = f"{maven_ere_folder}/{split}.jsonl"
        dataset = read_json_lines(split_path)
        for datapoint in dataset:
            sentences = datapoint["sentences"]
            if "events" not in datapoint:
                sentence_triggers.extend(
                    [
                        {"prompt": "triggers: " + sent, "response": "<unk>"}
                        for sent in sentences
                    ]
                )
                continue
            events = datapoint["events"]
            sent_id2mentions = {}
            for event in events:
                for mention in event["mention"]:
                    sent_id = mention["sent_id"]
                    if sent_id not in sent_id2mentions:
                        sent_id2mentions[sent_id] = []
                    sent_id2mentions[sent_id].append(
                        (mention["trigger_word"], mention["offset"][0])
                    )
            sent_id2mentions = sent_id2mentions.items()
            sent_id2mentions = sorted(sent_id2mentions, key=lambda x: x[0])
            sent_id2mentions = [
                (sent_id, sorted(mentions, key=lambda x: x[1]))
                for sent_id, mentions in sent_id2mentions
            ]
            input_outputs = [
                {
                    "prompt": "triggers: " + sentences[sent_id],
                    "response": " | ".join([mention[0] for mention in mentions]),
                }
                for sent_id, mentions in sent_id2mentions
            ]
            sentence_triggers.extend(input_outputs)

        # Save the dataset to the output_folder + split.csv path using to_csv
        output_path = f"{output_folder}/{split}.csv"
        pd.DataFrame(sentence_triggers).to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
