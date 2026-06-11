#!/usr/bin/env python3
"""Offline data prep for the 100M GPT benchmark (oracle-side; never at runtime).

Downloads a TinyStories text slice, trains a byte-level BPE tokenizer
(vocab 8192), and encodes the corpus to a flat little-endian uint16 token file
consumed by both the C++ trainer and the JAX baseline.

Outputs (in bench/data/):
  corpus.txt       raw text slice
  tokenizer.json   trained BPE tokenizer (HF tokenizers format)
  tokens.bin       uint16 token ids, little-endian
  meta.json        {vocab_size, num_tokens}

Run inside ~/venv-maxtext-py312 (needs `tokenizers`, `requests`).
"""
import json
import os
import sys

import requests
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
SLICE_BYTES = 100 * 1024 * 1024   # 100 MB of text
VOCAB = 8192

def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out, exist_ok=True)
    corpus = os.path.join(out, "corpus.txt")

    if not os.path.exists(corpus):
        print(f"downloading {SLICE_BYTES//2**20} MB of TinyStories...")
        r = requests.get(URL, stream=True, timeout=60)
        r.raise_for_status()
        got = 0
        with open(corpus, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got >= SLICE_BYTES:
                    break
        print(f"wrote {got} bytes")

    tok_path = os.path.join(out, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("training BPE tokenizer (vocab %d)..." % VOCAB)
        tok = Tokenizer(models.BPE(unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB, special_tokens=[], show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train([corpus], trainer)
        tok.save(tok_path)
    else:
        tok = Tokenizer.from_file(tok_path)

    print("encoding corpus...")
    import numpy as np
    ids_all = []
    with open(corpus, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(8 << 20)
            if not block:
                break
            ids_all.append(np.array(tok.encode(block).ids, dtype=np.uint16))
            sys.stdout.write("."); sys.stdout.flush()
    tokens = np.concatenate(ids_all)
    assert tokens.max() < VOCAB
    tokens.tofile(os.path.join(out, "tokens.bin"))
    meta = {"vocab_size": VOCAB, "num_tokens": int(tokens.size)}
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"\n{meta}")

if __name__ == "__main__":
    main()
