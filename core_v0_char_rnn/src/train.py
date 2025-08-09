# train.py — training CLI (with checkpoint that includes vocab chars)
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import set_seed, build_vocab, make_batches, CharRNN

SEED_TEXT = """SYSTEM: You are concise, clear, and polite. Answer in 1–2 sentences.
USER: hello
AI: Hello! How can I help you?
"""

def load_corpus(path: Path) -> str:
    if path.exists():
        txt = path.read_text(encoding="utf-8")
        src = str(path)
    else:
        txt, src = SEED_TEXT, "<SEED_TEXT>"
    if not txt.strip():
        txt, src = SEED_TEXT, "<SEED_TEXT>"
    print(f"[info] load corpus from: {src} | chars={len(txt)}")
    return txt

def main():
    here = Path(__file__).resolve().parent
    v0 = here.parent
    default_corpus = v0 / "data" / "corpus.txt"
    default_save = v0 / "checkpoints" / "char_rnn.pt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=str, default=str(default_corpus))
    ap.add_argument("--epochs", type=int, default=5 if not torch.cuda.is_available() else 3)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--save", type=str, default=str(default_save))
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    set_seed(0, fast=args.fast)

    corpus_path = Path(args.corpus)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(corpus_path)
    chars, stoi, itos = build_vocab(corpus)
    vocab = len(chars)

    seq_len = args.seq_len if args.seq_len is not None else min(96, max(32, len(corpus) // 3))
    X, Y = make_batches(corpus, stoi, seq_len=seq_len, step=1)
    if X is None:
        raise ValueError(f"Corpus is too short for SEQ_LEN={seq_len}.")

    ds = TensorDataset(X, Y)
    bs_auto = 64 if torch.cuda.is_available() else 32
    bs = args.batch if args.batch else max(1, min(bs_auto, len(ds)))
    loader = DataLoader(
        ds, batch_size=bs, shuffle=True, drop_last=False,
        num_workers=0, pin_memory=torch.cuda.is_available()
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharRNN(vocab_size=vocab, emb=args.emb, hid=args.hid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Device: {device} | Vocab: {vocab} | Batches: {len(loader)} | BS: {bs} | SEQ_LEN: {seq_len}")
    model.train()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, vocab), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs} loss {tot/max(1,n):.3f}")

    # save checkpoint with chars to reconstruct vocab at inference
    ckpt = {
        "state_dict": model.state_dict(),
        "chars": chars,
        "emb": args.emb,
        "hid": args.hid,
        "seq_len": seq_len,
    }
    torch.save(ckpt, save_path)
    print(f"[info] saved checkpoint to: {save_path.resolve()}")

if __name__ == "__main__":
    main()
