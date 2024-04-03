import requests
import typer


URL = "http://67.176.72.197:4040/predict/semantics"

app = typer.Typer()


def reformat(sentence, output_json):
    """
    Reformats SRL output into spacy-like output
    :param
        sentence: str
        output_json: output from jgung/verbnet-parser docker
    :return: [(['like', 'like.01', 2, 6], [('ARG0', 'I', 0, 1), ('ARG1', 'this example sentence', 7, 27)])]
    """
    srl_out = []
    for prop in output_json["props"]:
        predicate_sense = prop["sense"]
        predicate = ""
        predicate_begin, predicate_end = -1, -1
        arg_tuples = []
        found_predicate, found_args = False, False
        for span in prop["spans"]:
            if span["isPredicate"]:
                found_predicate = True
                predicate = span["text"]
                predicate_begin, predicate_end = wordoffsets2charoffsets(
                    sentence, span["start"], span["end"]
                )
            else:
                found_args = True
                arg = span["text"]
                arg_begin, arg_end = wordoffsets2charoffsets(
                    sentence, span["start"], span["end"]
                )
                arg_label = span["pb"]
                arg_tuples.append((arg_label, arg, arg_begin, arg_end))

        if found_predicate and found_args:
            srl_out.append(
                (
                    [predicate, predicate_sense, predicate_begin, predicate_end],
                    [arg_tuple for arg_tuple in arg_tuples],
                )
            )
    return srl_out


def wordoffsets2charoffsets(sentence, word_start, word_end):
    """
    to be fixed, this is a bit hacky
    :param sentence:
    :param word_start:
    :param word_end:
    :return:
    """
    whitespace_tokenized = sentence.split(" ")
    char_start = 0
    char_end = 0
    char_count = 0
    for i, token in enumerate(whitespace_tokenized):
        if i == word_start:
            char_start = char_count
        char_count += len(token)
        if i == word_end:
            char_end = char_count
        if i < len(whitespace_tokenized):
            # add one for whitespace
            char_count += 1

    return char_start, char_end


@app.command()
def jgung_srl(sentences, url=URL):
    """
    Generate SRL output for given sentences
    e.g.: sentences = ["I like this example sentence ."]
          srl_sentences = [(['like', 'like.01', 2, 6], [('ARG0', 'I', 0, 1), ('ARG1', 'this example sentence', 7, 27)])]
    :param
    sentences: List[str]
    url: str. Use the default URL provided. Or provide the locally run verbnet service url.
    :return:
    """
    srl_outputs = []
    for sentence in sentences:
        response = requests.post(url, data={"utterance": sentence})
        response_json = response.json()
        reformatted_output = reformat(sentence, response_json)
        srl_outputs.append(reformatted_output)
    return srl_outputs


if __name__ == "__main__":
    app()
