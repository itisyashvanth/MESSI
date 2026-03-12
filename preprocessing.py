"""
MESSI — Layer 1: Preprocessing
================================
Emoji-aware tokenizer + emoji vocabulary + combined embedding (ℝ^350).

Classes / functions exported:
  split_emoji_tokens(text)            → List[str]
  build_emoji_aware_nlp()             → Language
  tokenize(text, nlp)                 → List[str]
  build_vocab_from_texts(texts)       → Dict[str, int]
  save_vocab(vocab, path)
  load_vocab(path)                    → Dict[str, int]
  get_emoji_index(token, vocab)       → int
  CombinedEmbedding(emoji_vocab)      → nn.Module  (ℝ^350)
  EmbeddingExtractor(nlp, emoji_vocab)
  load_embedding_components(nlp)      → (CombinedEmbedding, EmbeddingExtractor)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import emoji as emoji_lib
import spacy
from spacy.language import Language

from config import (
    SPACY_MODEL, SPACY_VECTOR_DIM, EMOJI_EMBEDDING_DIM, EMBEDDING_DIM,
    UNK_TOKEN, PAD_TOKEN, EMOJI_VOCAB_PATH,
)


# ═══════════════════════════════════════════════════════════════
#  Tokenizer
# ═══════════════════════════════════════════════════════════════

def split_emoji_tokens(text: str) -> List[str]:
    """Split a string so every emoji character becomes its own token."""
    tokens: List[str] = []
    buf = ""
    for char in text:
        if char in emoji_lib.EMOJI_DATA:
            if buf.strip():
                tokens.extend(buf.strip().split())
            tokens.append(char)
            buf = ""
        else:
            buf += char
    if buf.strip():
        tokens.extend(buf.strip().split())
    return [t for t in tokens if t]


def build_emoji_aware_nlp() -> Language:
    """Load spaCy model with a custom emoji-splitting tokenizer."""
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        # Fallback to small model if md not installed
        nlp = spacy.load("en_core_web_sm")

    from spacy.tokenizer import Tokenizer

    original_tokenizer = nlp.tokenizer

    def custom_tokenize(text):
        words = split_emoji_tokens(text)
        spaces = [True] * len(words)
        return spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)

    nlp.tokenizer = custom_tokenize
    return nlp


def tokenize(text: str, nlp: Language) -> List[str]:
    """Tokenize text using the emoji-aware spaCy pipeline."""
    if not text.strip():
        return []
    doc = nlp(text)
    return [token.text for token in doc if token.text.strip()]


# ═══════════════════════════════════════════════════════════════
#  Emoji Vocabulary
# ═══════════════════════════════════════════════════════════════

def build_vocab_from_texts(texts) -> Dict[str, int]:
    """Scan texts/tokens for emoji and build {emoji: index} vocab."""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for item in texts:
        chars = item if isinstance(item, str) else " ".join(item)
        for char in chars:
            if char in emoji_lib.EMOJI_DATA and char not in vocab:
                vocab[char] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: Path = EMOJI_VOCAB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path = EMOJI_VOCAB_PATH) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_emoji_index(token: str, vocab: Dict[str, int]) -> int:
    return vocab.get(token, vocab.get(UNK_TOKEN, 1))


# ═══════════════════════════════════════════════════════════════
#  CombinedEmbedding: spaCy(300) + emoji(50) = ℝ^350
# ═══════════════════════════════════════════════════════════════

class CombinedEmbedding(nn.Module):
    """Concatenate frozen spaCy vector (300d) with trainable emoji embedding (50d)."""

    def __init__(
        self,
        emoji_vocab: Dict[str, int],
        emoji_emb_dim: int = EMOJI_EMBEDDING_DIM,
        spacy_vec_dim: int = SPACY_VECTOR_DIM,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.emoji_vocab   = emoji_vocab
        self.emoji_emb_dim = emoji_emb_dim
        self.spacy_vec_dim = spacy_vec_dim
        self.output_dim    = spacy_vec_dim + emoji_emb_dim  # 350

        self.emoji_embedding = nn.Embedding(
            num_embeddings=len(emoji_vocab),
            embedding_dim=emoji_emb_dim,
            padding_idx=padding_idx,
        )
        nn.init.xavier_uniform_(self.emoji_embedding.weight.data)
        self.emoji_embedding.weight.data[padding_idx].zero_()

    def forward(self, spacy_vectors: torch.Tensor, emoji_ids: torch.Tensor) -> torch.Tensor:
        emoji_vecs = self.emoji_embedding(emoji_ids)           # (B, L, 50)
        return torch.cat([spacy_vectors, emoji_vecs], dim=-1)  # (B, L, 350)


class EmbeddingExtractor:
    """Convert raw token lists → (spacy_tensor, emoji_tensor) pairs."""

    def __init__(self, nlp: Language, emoji_vocab: Dict[str, int]):
        self.nlp         = nlp
        self.emoji_vocab = emoji_vocab

    def extract(self, tokens: List[str]) -> Tuple[np.ndarray, List[int]]:
        vec_dim   = getattr(self.nlp.vocab, "vectors_length", None) or SPACY_VECTOR_DIM
        spacy_vecs = np.zeros((len(tokens), vec_dim), dtype=np.float32)
        emoji_ids: List[int] = []
        for i, tok in enumerate(tokens):
            if tok in emoji_lib.EMOJI_DATA:
                emoji_ids.append(get_emoji_index(tok, self.emoji_vocab))
            else:
                lex = self.nlp.vocab[tok]
                if lex.has_vector:
                    spacy_vecs[i] = lex.vector
                emoji_ids.append(0)
        return spacy_vecs, emoji_ids

    def batch_extract(
        self,
        batch_tokens: List[List[str]],
        max_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_len is None:
            max_len = max(len(t) for t in batch_tokens)
        vec_dim = getattr(self.nlp.vocab, "vectors_length", None) or SPACY_VECTOR_DIM
        B = len(batch_tokens)
        batch_spacy = np.zeros((B, max_len, vec_dim), dtype=np.float32)
        batch_emoji = np.zeros((B, max_len), dtype=np.int64)
        for b, tokens in enumerate(batch_tokens):
            vecs, ids = self.extract(tokens[:max_len])
            L = len(tokens[:max_len])
            batch_spacy[b, :L] = vecs
            batch_emoji[b, :L] = ids
        return (
            torch.tensor(batch_spacy, dtype=torch.float32),
            torch.tensor(batch_emoji, dtype=torch.long),
        )


def load_embedding_components(nlp: Language) -> Tuple[CombinedEmbedding, EmbeddingExtractor]:
    if EMOJI_VOCAB_PATH.exists():
        vocab = load_vocab()
    else:
        vocab = build_vocab_from_texts(["😠", "😤", "😡", "💀", "🔥", "😊", "👍"])
        save_vocab(vocab)
    return CombinedEmbedding(emoji_vocab=vocab), EmbeddingExtractor(nlp, vocab)
