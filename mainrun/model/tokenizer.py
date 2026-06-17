from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders


def train_tokenizer(token_type: str, titles: list[str], vocab_size: int,
                    unk_token: str = "<unk>", pad_token: str = "<pad>",
                    eos_token: str = "<eos>") -> HFTokenizer:
    """Train a subword tokenizer of `token_type` on `titles`. All variants share a ByteLevel
    pre-tokenizer + decoder so they operate on the same byte alphabet -> a fair vocab comparison."""
    specials = [pad_token, eos_token, unk_token]

    if token_type == "bpe":
        tk = HFTokenizer(models.BPE(unk_token=unk_token))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel()
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials)

    # https://huggingface.co/docs/tokenizers/en/api/trainers#tokenizers.trainers.UnigramTrainer
    elif token_type == "unigram":
        tk = HFTokenizer(models.Unigram())                 # unk_id is set by the trainer below
        tk.pre_tokenizer = pre_tokenizers.ByteLevel()
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=specials,
                                          unk_token=unk_token)

    # https://huggingface.co/docs/tokenizers/en/api/trainers#tokenizers.trainers.WordPieceTrainer
    elif token_type == "wordpiece":
        tk = HFTokenizer(models.WordPiece(unk_token=unk_token, max_input_chars_per_word=100))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel()
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=specials)

    else:
        raise ValueError(f"unknown token_type {token_type!r} (expected: bpe | unigram | wordpiece)")

    tk.train_from_iterator(titles, trainer)
    return tk


class Tokenizer:
    def __init__(self, tokenizer: HFTokenizer):
        self.tk = tokenizer
        self.stoi = {tok: i for tok, i in tokenizer.get_vocab().items()}
        self.itos = {i: tok for tok, i in tokenizer.get_vocab().items()}

    def encode(self, s: str) -> list[int]:
        return self.tk.encode(s).ids

    def decode(self, ids: list[int]) -> str:
        return self.tk.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        return self.tk.get_vocab_size()
