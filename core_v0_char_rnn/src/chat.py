# chat.py — inference CLI with RAG-lite first, CharRNN fallback
from pathlib import Path
import argparse

import torch

from model import (
    set_seed, vocab_from_chars, CharRNN,
    parse_pairs, retrieve_answer, rare_ids_from_text
)

def load_corpus(path: Path) -> str:
    if path.exists():
        txt = path.read_text(encoding="utf-8")
        return txt if txt.strip() else ""
    return ""

def main():
    here = Path(__file__).resolve().parent
    v0 = here.parent
    default_corpus = v0 / "data" / "corpus.txt"

    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, required=True, help="Path to checkpoint .pt")
    ap.add_argument("--corpus", type=str, default=str(default_corpus))
    ap.add_argument("--temp", type=float, default=0.4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new", type=int, default=180)
    ap.add_argument("--min_sim", type=float, default=0.62, help="RAG similarity threshold")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--greedy", action="store_true", help="Force greedy decoding (temp=0, top_p=0)")
    args = ap.parse_args()

    set_seed(0, fast=args.fast)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load checkpoint
    ckpt_path = Path(args.load)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    chars = ckpt["chars"]
    emb = ckpt.get("emb", 64)
    hid = ckpt.get("hid", 256)

    # vocab + model
    stoi, itos = vocab_from_chars(chars)
    model = CharRNN(vocab_size=len(chars), emb=emb, hid=hid).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # RAG-lite over current corpus
    corpus_path = Path(args.corpus)
    corpus = load_corpus(corpus_path)
    qa_pairs = parse_pairs(corpus) if corpus else []
    rare_ids = rare_ids_from_text(corpus, stoi, min_count=3).to(device) if corpus else torch.empty(0, dtype=torch.long, device=device)

    print("RNN ready. Type your question; 'exit' to quit.\n")
    SYSTEM = "SYSTEM: You are concise and clear.\n"

    try:
        while True:
            try:
                q = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"exit", "quit"}:
                break

            # 1) RAG-lite exact-ish answer if similar enough
            ans_ret = retrieve_answer(q, qa_pairs, min_sim=args.min_sim) if qa_pairs else None
            if ans_ret:
                print(f"AI> {ans_ret}\n")
                continue

            # 2) Generative fallback
            prompt = SYSTEM + f"USER: {q}\nAI: "
            temp = 0.0 if args.greedy else args.temp
            top_p = 0.0 if args.greedy else args.top_p
            with torch.no_grad():
                ans = model.generate(
                    prefix=prompt,
                    stoi=stoi,
                    itos=itos,
                    max_new=args.max_new,
                    temperature=temp,
                    top_p=top_p,
                    newline_penalty=0.4,
                    repetition_penalty=1.4,
                    last_n=24,
                    min_len_before_break=12,
                    rare_ids=rare_ids if rare_ids.numel() > 0 else None,
                )
            one_line = (ans.split("\n")[0] if "\n" in ans else ans).strip() or "Okay."
            print(f"AI> {one_line}\n")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
