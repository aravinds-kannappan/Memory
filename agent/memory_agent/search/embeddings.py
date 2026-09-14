from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

# Lightweight local embedding: hashed bag-of-words + character n-grams.
# Swap for sentence-transformers via MEMORY_EMBEDDING=sentence later.

DIM = 384
_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _token_re.findall(text.lower())


def embed_text(text: str, dim: int = DIM) -> list[float]:
    tokens = tokenize(text)
    if not tokens:
        return [0.0] * dim
    vec = np.zeros(dim, dtype=np.float64)
    # unigrams
    for tok in tokens:
        h = hash(tok) % dim
        vec[h] += 1.0
    # bigrams
    for a, b in zip(tokens, tokens[1:]):
        h = hash(f"{a}_{b}") % dim
        vec[h] += 0.5
    # char trigrams for fuzzy recall ("minimum order" ≈ "min order")
    compact = "".join(tokens)[:2000]
    for i in range(max(0, len(compact) - 2)):
        gram = compact[i : i + 3]
        h = hash(f"c:{gram}") % dim
        vec[h] += 0.15
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def semantic_rank(
    query: str, candidates: Iterable[dict], limit: int = 20
) -> list[dict]:
    q = embed_text(query)
    scored = []
    for c in candidates:
        emb = c.get("embedding")
        if not emb:
            continue
        score = cosine(q, emb)
        item = dict(c)
        item["semantic_score"] = score
        scored.append(item)
    scored.sort(key=lambda x: x["semantic_score"], reverse=True)
    return scored[:limit]
