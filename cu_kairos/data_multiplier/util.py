import json


class JsonParser:
    def __init__(self):
        pass

    def parse(self, output):
        return json.loads(output)


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
