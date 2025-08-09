# model.py — core components (CharRNN + vocab + batching + RAG-lite + generation)
import string, random, difflib
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ---------- utils ----------
def set_seed(seed: int = 0, fast: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not fast
    torch.backends.cudnn.benchmark = fast


def build_vocab(text: str):
    base = string.ascii_letters + string.digits + string.punctuation + " \t\n"
    chars = sorted(set(text).union(set(base)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return chars, stoi, itos


def vocab_from_chars(chars: List[str]):
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(s: str, stoi: dict, fallback: str = " ") -> torch.Tensor:
    return torch.tensor([stoi.get(c, stoi[fallback]) for c in s], dtype=torch.long)


def decode(t: torch.Tensor, itos: dict) -> str:
    return "".join(itos[int(i)] for i in t)


def make_batches(text: str, stoi: dict, seq_len: int = 64, step: int = 1):
    ids = encode(text, stoi)
    X, Y = [], []
    for i in range(0, len(ids) - seq_len - 1, step):
        X.append(ids[i:i + seq_len])
        Y.append(ids[i + 1:i + seq_len + 1])
    if not X:
        return None, None
    return torch.stack(X), torch.stack(Y)


# ---------- RAG-lite (string similarity over Q/A) ----------
def parse_pairs(text: str) -> List[Tuple[str, str]]:
    """
    Expects alternating lines:
    USER: ...
    AI:   ...
    """
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
    best = max(qa_pairs, key=lambda qa: difflib.SequenceMatcher(None, qn, qa[0]).ratio())
    sim = difflib.SequenceMatcher(None, qn, best[0]).ratio()
    return best[1] if sim >= min_sim else None


def rare_ids_from_text(text: str, stoi: dict, min_count: int = 3) -> torch.Tensor:
    from collections import Counter
    cnt = Counter(text)
    ids = [stoi[c] for c, n in cnt.items() if n < min_count and c in stoi]
    return torch.tensor(ids, dtype=torch.long)


# ---------- model ----------
class CharRNN(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 64, hid: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb)
        self.rnn = nn.RNN(emb, hid, num_layers=1, nonlinearity="tanh", batch_first=True)
        self.head = nn.Linear(hid, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)          # [B,T,E]
        out, h = self.rnn(e, h)    # [B,T,H]
        logits = self.head(out)    # [B,T,V]
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
        dev = next(self.parameters()).device
        ids = encode(prefix, stoi).to(dev)
        if ids.numel() == 0:
            ids = torch.tensor([stoi.get(" ", 0)], device=dev)

        # Prefill hidden state once with the whole prefix
        h = None
        seq = ids.view(1, -1)
        for tok in seq.split(1, dim=1):
            _, h = self.forward(tok, h)
        x = seq[:, -1:]

        out_ids: List[int] = []
        nl_id = stoi.get("\n", None)
        if rare_ids is None:
            rare_ids = torch.empty(0, dtype=torch.long, device=dev)
        else:
            rare_ids = rare_ids.to(dev)

        for _ in range(max_new):
            logits, h = self.forward(x, h)

            # scale by temperature (if temperature==0, we will do greedy)
            if temperature > 0:
                logits = logits / max(1e-6, temperature)

            # discourage newline too early
            if nl_id is not None:
                if len(out_ids) < min_len_before_break:
                    logits[0, nl_id] -= 5.0
                else:
                    logits[0, nl_id] -= newline_penalty

            # ban rare chars (optional)
            if rare_ids.numel() > 0:
                logits[0, rare_ids] -= 10.0

            # repetition penalty: downweight recently generated ids
            if out_ids:
                recent = torch.tensor(out_ids[-last_n:], device=dev, dtype=torch.long)
                logits[0, recent] -= repetition_penalty

            # choose next id: greedy if temp<=0 or top_p<=0, else nucleus
            if temperature <= 0 or top_p <= 0:
                next_id = torch.argmax(logits[:, -1, :], dim=-1)[0]
            else:
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                sp, si = torch.sort(probs[0], descending=True)
                csum = torch.cumsum(sp, dim=-1)
                k = int(torch.searchsorted(csum, torch.tensor(top_p, device=sp.device)).item()) + 1
                k = max(1, min(k, sp.numel()))
                p = sp[:k] / sp[:k].sum()
                idx = si[:k]
                next_id = idx[torch.multinomial(p, 1)]

            out_ids.append(int(next_id))
            x = next_id.view(1, 1)

            if nl_id is not None and int(next_id) == nl_id and len(out_ids) >= min_len_before_break:
                break

        return "".join(itos[i] for i in out_ids)
