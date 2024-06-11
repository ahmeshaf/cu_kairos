from cu_kairos import semantic_role_labeler_seq2seq

if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print(
        semantic_role_labeler_seq2seq(
            sentences,
            trigger_model_name="ahmeshaf/ecb_tagger_seq2seq",
            is_trigger_peft=False,
            srl_model="cu-kairos/propbank_srl_seq2seq_t5_large",
            is_srl_peft=False,
            batch_size=2,
        )
    )

    print(
        semantic_role_labeler_seq2seq(
            sentences,
            trigger_model_name="cu-kairos/flan-srl-large-peft",
            is_trigger_peft=True,
            srl_model="cu-kairos/flan-srl-large-peft",
            is_srl_peft=True,
            batch_size=2,
        )
    )