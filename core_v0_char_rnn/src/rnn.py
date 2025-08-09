# core_v0_char_rnn/src/rnn.py
# Character-Level RNN Chatbot (with RAG-lite retrieval)
# - Load corpus: data/corpus.txt (fallback ke SEED_TEXT)
# - Vocab aman (BASE_CHARS ∪ korpus), OOV -> spasi
# - Training stabil (clip grad, step=1, SEQ_LEN adaptif)
# - Generate: prefill h sekali, nucleus top-p, temperature,
#             repetition penalty, ban karakter langka, newline penalty
# - RAG-lite: cari jawaban mirip dari korpus (difflib) sebelum generatif
# - CLI arg: epochs, lr, batch, seq_len, temp, top_p, dll

import os, sys, math, time, random, string, argparse, difflib
from collections import Counter
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

# ---------------- utils & config ----------------
def set_seed(seed: int = 0, fast: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministik default; jika --fast, izinkan autotune
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.benchmark = fast

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED_TEXT = """SYSTEM: Kamu asisten singkat, jelas, sopan. Jawab 1–2 kalimat.
USER: halo
AI: Halo! Ada yang bisa saya bantu?
"""

def load_corpus(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        src = path
    else:
        txt, src = SEED_TEXT, "<SEED_TEXT>"
    if not txt.strip():
        txt, src = SEED_TEXT, "<SEED_TEXT>"
    print(f"[info] load corpus from: {src} | chars={len(txt)}")
    return txt

def build_vocab(text: str):
    BASE_CHARS = string.ascii_letters + string.digits + string.punctuation + " \t\n"
    chars = sorted(set(text).union(set(BASE_CHARS)))
    stoi = {c:i for i,c in enumerate(chars)}
    itos = {i:c for c,i in stoi.items()}
    return chars, stoi, itos

def encode(s: str, stoi: dict, fallback: str = " ") -> torch.Tensor:
    return torch.tensor([stoi.get(c, stoi[fallback]) for c in s], dtype=torch.long)

def decode(t: torch.Tensor, itos: dict) -> str:
    return "".join(itos[int(i)] for i in t)

def make_batches(text: str, stoi: dict, seq_len: int = 64, step: int = 1):
    ids = encode(text, stoi)
    X, Y = [], []
    for i in range(0, len(ids) - seq_len - 1, step):
        X.append(ids[i:i+seq_len])
        Y.append(ids[i+1:i+seq_len+1])
    if not X:
        return None, None
    return torch.stack(X), torch.stack(Y)

# ---------------- simple retrieval from corpus ----------------
def parse_pairs(text: str) -> List[Tuple[str, str]]:
    pairs, q = [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("USER:"):
            q = line[5:].strip().lower()
        elif line.startswith("AI:") and q is not None:
            a = line[3:].strip()
            pairs.append((q, a))
            q = None
    return pairs

def retrieve_answer(query: str, qa_pairs: List[Tuple[str, str]], min_sim: float = 0.62) -> Optional[str]:
    if not qa_pairs or not query.strip():
        return None
    qn = query.strip().lower()
    best_a, best_sim = None, -1.0
    for q, a in qa_pairs:
        sim = difflib.SequenceMatcher(None, qn, q).ratio()
        if sim > best_sim:
            best_sim, best_a = sim, a
    return best_a if best_sim >= min_sim else None

# ---------------- model ----------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.rnn = nn.RNN(emb, hid, num_layers=1, nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(hid, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)         # [B,T,E]
        out, h = self.rnn(e, h)   # [B,T,H]
        logits = self.head(out)   # [B,T,V]
        return logits, h

    @torch.no_grad()
    def generate(
        self,
        prefix: str,
        stoi: dict,
        itos: dict,
        max_new: int = 180,
        temperature: float = 0.4,
        top_p: float = 0.9,
        newline_penalty: float = 0.5,
        repetition_penalty: float = 1.4,
        last_n: int = 24,
        min_len_before_break: int = 12,
        rare_ids: Optional[torch.Tensor] = None,
    ) -> str:
        self.eval()
        ids = encode(prefix, stoi).to(DEVICE)
        if ids.numel() == 0:
            ids = torch.tensor([stoi.get(" ", 0)], device=DEVICE)
        # Prefill hidden state sekali
        h = None
        seq = ids.view(1, -1)
        for tok in seq.split(1, dim=1):
            _, h = self.forward(tok, h)
        x = seq[:, -1:]
        out_ids: List[int] = []
        nl_id = stoi.get("\n", None)
        if rare_ids is None:
            rare_ids = torch.empty(0, dtype=torch.long, device=DEVICE)
        else:
            rare_ids = rare_ids.to(DEVICE)

        for _ in range(max_new):
            logits, h = self.forward(x, h)
            logits = logits[:, -1, :] / max(1e-6, temperature)

            # Kurangi peluang newline (agar tak langsung putus)
            if nl_id is not None:
                if len(out_ids) < min_len_before_break:
                    logits[0, nl_id] -= 5.0
                else:
                    logits[0, nl_id] -= newline_penalty

            # Ban karakter langka
            if rare_ids.numel() > 0:
                logits[0, rare_ids] -= 10.0

            # Penalti pengulangan kasar: token terakhir N
            if out_ids:
                recent = torch.tensor(out_ids[-last_n:], device=DEVICE, dtype=torch.long)
                logits[0, recent] -= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            # Nucleus (top-p)
            sp, si = torch.sort(probs[0], descending=True)
            csum = torch.cumsum(sp, dim=-1)
            k = int(torch.searchsorted(csum, torch.tensor(top_p, device=sp.device)).item()) + 1
            k = max(1, min(k, sp.numel()))
            p = sp[:k] / sp[:k].sum()
            idx = si[:k]

            next_id = idx[torch.multinomial(p, 1)]
            out_ids.append(int(next_id))
            x = next_id.view(1,1)

            if nl_id is not None and int(next_id) == nl_id and len(out_ids) >= min_len_before_break:
                break

        return "".join(itos[i] for i in out_ids)

# ---------------- train & chat ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="data/corpus.txt")
    parser.add_argument("--epochs", type=int, default=5 if DEVICE=="cpu" else 3)
    parser.add_argument("--seq_len", type=int, default=None)  # auto jika None
    parser.add_argument("--batch", type=int, default=None)    # auto jika None
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--save", type=str, default="checkpoints/char_rnn.pt")
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--temp", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_sim", type=float, default=0.62)  # threshold retrieval
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="enable cuDNN autotune (nondeterministic)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    set_seed(0, fast=args.fast)

    CORPUS = load_corpus(args.corpus)
    QA_PAIRS = parse_pairs(CORPUS)
    chars, stoi, itos = build_vocab(CORPUS)
    vocab = len(chars)

    # Karakter langka (muncul < 3 kali)
    cnt = Counter(CORPUS)
    rare_ids = torch.tensor([stoi[c] for c, n in cnt.items() if n < 3], dtype=torch.long)

    # SEQ_LEN adaptif
    SEQ_LEN = args.seq_len if args.seq_len is not None else min(96, max(32, len(CORPUS)//3))
    X, Y = make_batches(CORPUS, stoi, seq_len=SEQ_LEN, step=1)
    if X is None:
        raise ValueError(f"Corpus terlalu pendek untuk SEQ_LEN={SEQ_LEN}.")

    train_ds = torch.utils.data.TensorDataset(X, Y)
    bs = args.batch if args.batch else max(1, min(64 if DEVICE=="cuda" else 32, len(train_ds)))
    pin = (DEVICE == "cuda")
    loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=False,
                                         num_workers=args.workers, pin_memory=pin)

    # Model
    model = CharRNN(vocab_size=vocab, emb=args.emb, hid=args.hid).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Optional load
    if args.load and os.path.exists(args.load):
        model.load_state_dict(torch.load(args.load, map_location=DEVICE))
        print(f"[info] loaded checkpoint: {args.load}")

    print(f"Device: {DEVICE} | Vocab: {vocab} | Batches: {len(loader)} | BS: {bs} | SEQ_LEN: {SEQ_LEN}")

    # Train
    model.train()
    for ep in range(args.epochs):
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(xb)
            loss = loss_fn(logits.reshape(-1, vocab), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep+1}/{args.epochs} loss {tot/max(1,n):.3f}")

    # Save
    try:
        torch.save(model.state_dict(), args.save)
        print(f"[info] saved checkpoint to: {args.save}")
    except Exception as e:
        print(f"[warn] save failed: {e}", file=sys.stderr)

    # Chat
    print("\nRNN siap. Ketik pertanyaan; 'exit' untuk keluar.\n")
    SYSTEM = "SYSTEM: Kamu asisten singkat, jelas.\n"
    model.eval()
    try:
        while True:
            try:
                q = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q or q.lower() in {"exit","quit"}:
                break

            # 1) Retrieval dulu: jika mirip, jawab persis
            ans_ret = retrieve_answer(q, QA_PAIRS, min_sim=args.min_sim)
            if ans_ret:
                print(f"AI> {ans_ret}\n")
                continue

            # 2) Generatif fallback
            prompt = SYSTEM + f"USER: {q}\nAI: "
            with torch.no_grad():
                ans = model.generate(
                    prefix=prompt,
                    stoi=stoi,
                    itos=itos,
                    max_new=180,
                    temperature=args.temp,
                    top_p=args.top_p,
                    newline_penalty=0.4,
                    repetition_penalty=1.4,
                    last_n=24,
                    min_len_before_break=12,
                    rare_ids=rare_ids,
                )
            ans_line = (ans.split("\n")[0] if "\n" in ans else ans).strip()
            if not ans_line:
                ans_line = "Baik."
            print(f"AI> {ans_line}\n")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
