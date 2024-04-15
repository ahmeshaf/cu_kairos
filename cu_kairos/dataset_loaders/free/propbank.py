import os.path
import pickle

import requests
import typer
import zipfile

from bs4 import BeautifulSoup
from glob import glob
from pathlib import Path
from tqdm import tqdm

from ...util import ensure_directory, to_csv

PB_GIT = "https://github.com/propbank/propbank-frames.git"

app = typer.Typer()


def download_git_archive(git_repo, lexicon_dir):
    ensure_directory(lexicon_dir)

    repo_name = git_repo.split(".git")[0].split("/")[-1]
    archive_link = git_repo.split(".git")[0] + "/archive/master.zip"

    response = requests.get(archive_link, allow_redirects=True)

    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kb
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc="downloading FrameSet from GitHub",
    )
    if not os.path.exists(lexicon_dir + f"/{repo_name}/"):
        os.makedirs(lexicon_dir + f"/{repo_name}/")

    archive_file_path = lexicon_dir + f"/{repo_name}/master.zip"

    with open(archive_file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    with zipfile.ZipFile(archive_file_path, "r") as zip_ref:
        zip_ref.extractall(lexicon_dir + f"/{repo_name}/")

    return list(glob(lexicon_dir + f"/{repo_name}/*/frames/"))[0]


def _get_eg_arg_spans(eg_):
    eg_text = eg_.find("text").text
    if (
        eg_.find("propbank")
        and eg_.find("propbank").find("rel")
        and eg_.find("propbank").find("rel")["relloc"] != "?"
    ):
        t_text = eg_.find("propbank").find("rel").text
        t_indices = [int(i) for i in eg_.find("propbank").find("rel")["relloc"].split()]
        t_start = t_indices[0]
        t_end = t_indices[-1]

        if t_start > t_end:
            return None

        args = eg_.find("propbank").find_all("arg")
        arg_token_map = []
        for a in args:
            try:
                a_map = {
                    "text": a.text,
                    "label": a["type"].replace("ARG", "ARG-"),
                    "token_start": int(a["start"]),
                    "token_end": int(a["end"]),
                }
                if int(a["start"]) <= int(a["end"]):
                    arg_token_map.append(a_map)
            except ValueError:
                pass

        return {
            "src": eg_["src"].replace("ontonotes ", "ontonotes/"),
            "text": eg_text,
            "head_span": {
                "label": "V",
                "text": t_text,
                "token_start": t_start,
                "token_end": t_end,
            },
            "relations": arg_token_map,
        }
    else:
        return None


def get_pb_dict(
    frames_folder: Path,
):
    print(frames_folder)
    frame_files = list(glob(str(frames_folder) + "/*.xml"))
    print(len(frame_files))
    pb_dict = {}
    for frame in tqdm(frame_files, desc="Reading FrameSet"):
        with open(frame) as ff:
            frame_bs = BeautifulSoup(ff.read(), parser="lxml", features="lxml")
            predicate = frame_bs.find("predicate")["lemma"]
            rolesets = frame_bs.find_all("roleset")
            for roleset in rolesets:
                rs_id = roleset["id"]
                lexlinks = [dict(l.attrs) for l in roleset.find_all("lexlink") if l]
                rs_defs = {
                    "sense": rs_id,
                    "lemma": predicate,
                    "frame": predicate,
                    "definition": roleset["name"],
                    "aliases": [al.text for al in roleset.find_all("alias")],
                    "lexlinks": lexlinks,
                }
                if roleset.find("roles"):
                    rs_defs["roles"] = [
                        {"id": "ARG-" + r["n"], "definition": r["descr"]}
                        for r in roleset.find("roles").find_all("role")
                    ]
                rs_examples = [
                    _get_eg_arg_spans(eg_) for eg_ in roleset.find_all("example")
                ]
                rs_defs["examples"] = rs_examples
                pb_dict[rs_id] = rs_defs

    return pb_dict


@app.command()
def create_trigger_dataset(
    lexicon_out_folder: Path, dataset_folder: Path, force: bool = False
):
    ensure_directory(dataset_folder)
    pb_dict_file_path = lexicon_out_folder / f"pb.dict"

    if not pb_dict_file_path.exists() or force:
        frame_set_folder = download_git_archive(PB_GIT, str(lexicon_out_folder))
        pb_dict = get_pb_dict(frame_set_folder)
        pickle.dump(pb_dict, open(pb_dict_file_path, "wb"))

    pb_dict = pickle.load(open(pb_dict_file_path, "rb"))

    t_examples = [eg_ for roleset in pb_dict.values() for eg_ in roleset["examples"]]

    tasks = [(eg_["text"], eg_["head_span"]["text"]) for eg_ in t_examples if eg_ and len(eg_["text"].split()) < 10]

    datapoints = [{
        "sentence_id": str(i),
        "sentence": sentence,
        "entities": "",
        "triggers": f"{trigger}",

    } for i, (sentence, trigger) in enumerate(tasks)]

    to_csv(datapoints, str(dataset_folder) + f"/train.csv")


if __name__ == "__main__":
    app()

# pb_git = 'https://github.com/propbank/propbank-frames.git'
# frame_set_folder = download_git_archive(pb_git, './lexicons/')
# parse_frames(frame_set_folder, './propbank/')
