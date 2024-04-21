from cu_kairos.srl import jgung_srl

sentences = ["Local fuel prices have also largely recovered since that shock, making it cheaper to transport food than a couple of months ago ."]
srl_out = jgung_srl(sentences)

print(srl_out[0][-1])