from cu_kairos.srl import jgung_srl

# sentences = ["Local fuel prices have also largely recovered since that shock, making it cheaper to transport food than a couple of months ago ."]
sentences = ["Local fuel prices have also largely recovered since that shock, making it cheaper to transport food than a couple of months ago .",
    "Few will ever get to walk the icy plains of Antarctica - 5FM's Nick Hamman tells us what it was like By Nikita Coetzee At the southernmost tip of the globe, covered in pristine white snow that stretches as far as the eye can see, pierced only by seemingly insurmountable walls of ice, is Antarctica."]
srl_out = jgung_srl(sentences)

for sentence, srl_ in zip(sentences, srl_out):
    for predicate, args in srl_:
        predicate_text = predicate[0]
        print(predicate_text)
        predicted_pred = sentence[predicate[2]:predicate[3]]
        assert predicted_pred == predicate_text # sanity check
