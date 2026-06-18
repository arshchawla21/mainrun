import os
import tempfile
from contextlib import contextmanager

from tokenizers import Tokenizer as HFTokenizer, models, trainers, pre_tokenizers, decoders, Regex


@contextmanager
def _chdir(path):
    """Run a block with cwd=path, always restoring the previous cwd. The superbpe fork decides
    'resume vs train-from-scratch' by looking for a merges.txt in the *cwd*, so the two SuperBPE
    stages are driven purely by controlling cwd around each .train() call."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _train_superbpe(titles: list[str], vocab_size: int, specials: list[str],
                    transition_ratio: float) -> HFTokenizer:
    """SuperBPE (Liu et al. 2025, arXiv:2503.13423) via the tokenizers-superbpe fork.

    Stage 1 (subwords): regular byte-level BPE *with* whitespace pre-tokenization, up to a
        transition point t = round(transition_ratio * vocab_size). cwd is an empty temp dir,
        so the fork's BpeTrainer runs do_train_original() (train from scratch).
    Stage 2 (superwords): resume from the stage-1 merges *without* whitespace pre-tokenization,
        up to the final vocab_size. We drop stage-1's merges.txt into cwd, so the fork runs
        do_train_extend() -> it inherits those merges, then learns new ones that may span spaces.

    At transition_ratio=1.0 this reduces exactly to the plain `bpe` path (stage 2 is a no-op)."""
    t = round(transition_ratio * vocab_size)
    t = max(len(specials) + 257, min(t, vocab_size - 1))   # leave room for the byte alphabet & keep t < final
    with tempfile.TemporaryDirectory() as d:
        # --- stage 1: subwords. ByteLevel(use_regex=True) splits on whitespace -> no cross-word merges.
        with _chdir(d):
            s1 = HFTokenizer(models.BPE())
            s1.pre_tokenizer = pre_tokenizers.ByteLevel()
            s1.decoder = decoders.ByteLevel()
            s1.train_from_iterator(titles, trainers.BpeTrainer(vocab_size=t, special_tokens=specials))
        s1.model.save(d)                                   # writes vocab.json + merges.txt into d

        # --- stage 2: superwords. ByteLevel(use_regex=False) does NOT split -> merges may cross spaces.
        with _chdir(d):                                    # merges.txt now present -> fork extends
            s2 = HFTokenizer(models.BPE())
            s2.pre_tokenizer = pre_tokenizers.ByteLevel(use_regex=False)
            s2.decoder = decoders.ByteLevel()
            s2.train_from_iterator(titles, trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=specials))

    # the final pre-tokenizer must stay whitespace-free, or the superwords can never form at encode time.
    s2.pre_tokenizer = pre_tokenizers.ByteLevel(use_regex=False)
    return s2


def _build_wordlevel(titles: list[str], pad_token: str = "<pad>",
                     eos_token: str = "<eos>") -> HFTokenizer:
    """Word-level tokenizer: no training, no vocab cap, no <unk>. Splitting rule: whitespace
    delimits, word-connecting punctuation ':' '/' '-' stays attached, and every other punctuation
    char becomes its own token. So 'HN:', 'open-source', 'York/Remote' stay whole, while
    'tool,' -> 'tool' ',' and 'really?' -> 'really' '?'. The vocab is built by running this exact
    pre-tokenizer over the FULL (train+val) corpus, so it covers everything that will ever be
    encoded -> zero OOV, hence no <unk> needed. No ByteLevel decoder -> WordLevel decode space-joins.
    To keep more connectors (e.g. '_' '+' '.'), add them inside the regex character class."""
    # 1) WhitespaceSplit drops whitespace; 2) isolate any char that isn't a letter/digit/connector
    #    (connectors kept: ':' '/' '-'  -- '-' is last in the class so it's a literal, not a range).
    pre = pre_tokenizers.Sequence([
        pre_tokenizers.WhitespaceSplit(),
        pre_tokenizers.Split(pattern=Regex(r"[^\p{L}\p{N}:/-]"), behavior="isolated"),
    ])
    specials = [pad_token, eos_token]                  # keep eos at id 1 (matches the other tokenizers)
    vocab = {tok: i for i, tok in enumerate(specials)}
    for t in titles:
        for w, _ in pre.pre_tokenize_str(t):           # build vocab with the SAME rule used at encode
            if w not in vocab:
                vocab[w] = len(vocab)
    tk = HFTokenizer(models.WordLevel(vocab=vocab))    # no unk_token -> truly no <unk>
    tk.pre_tokenizer = pre
    tk.add_special_tokens(specials)                    # so a glued '<eos>' (title<eos>title) is extracted
    return tk


def train_tokenizer(token_type: str, titles: list[str], vocab_size: int,
                    unk_token: str = "<unk>", pad_token: str = "<pad>",
                    eos_token: str = "<eos>", transition_ratio: float = 0.75) -> HFTokenizer:
    """Train a subword tokenizer of `token_type` on `titles`. All variants share a ByteLevel
    pre-tokenizer + decoder so they operate on the same byte alphabet -> a fair vocab comparison.
    `transition_ratio` is used only by token_type='superbpe' (fraction of vocab spent on stage-1 subwords)."""
    specials = [pad_token, eos_token, unk_token]

    # https://github.com/PythonNut/superbpe  /  https://github.com/alisawuffles/tokenizers-superbpe
    # the rust `tokenizers` module is patched with the superbpe fork to enable two-stage training.
    if token_type == "superbpe":
        return _train_superbpe(titles, vocab_size, specials, transition_ratio)

    # direct word-level: no training, no <unk>, no vocab_size = len(set([t.split() for t in titles]))
    if token_type == "wordlevel":
        return _build_wordlevel(titles, pad_token, eos_token)

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
        raise ValueError(f"unknown token_type {token_type!r} "
                         f"(expected: bpe | unigram | wordpiece | superbpe | wordlevel)")

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
