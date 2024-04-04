from cu_kairos.evt_tagger import ecb_tagger

sentences = [
    "I like this sentence and hate this sentence and I like this thing",
    "The earthquake took 10 lives .",
]

triggers = ecb_tagger(sentences)

print(triggers)
