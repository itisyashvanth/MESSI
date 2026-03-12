"""Tests for Layer 1: Emoji-Aware Tokenizer"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from layers.layer1_preprocessing.tokenizer import split_emoji_tokens, build_emoji_aware_nlp, tokenize
from layers.layer1_preprocessing.emoji_vocab import build_vocab_from_texts, get_emoji_index


class TestEmojiSplitter:
    def test_splits_adjacent_emoji(self):
        tokens = split_emoji_tokens("delayed😤")
        assert "😤" in tokens
        assert "delayed" in tokens

    def test_keeps_plain_text_intact(self):
        tokens = split_emoji_tokens("hello world")
        assert tokens == ["hello", "world"]

    def test_multiple_emojis(self):
        tokens = split_emoji_tokens("fail💀🔥")
        assert "💀" in tokens
        assert "🔥" in tokens

    def test_no_empty_tokens(self):
        tokens = split_emoji_tokens("  😠  ")
        assert all(t.strip() != "" for t in tokens)

    def test_order_hash_id(self):
        tokens = split_emoji_tokens("order #782 not delivered 😠")
        assert "😠" in tokens
        assert "782" in tokens or "#782" in tokens

    def test_no_emoji_text_unchanged(self):
        text = "normal sentence without emojis"
        tokens = split_emoji_tokens(text)
        assert len(tokens) > 0
        assert "😠" not in tokens


class TestEmojiVocab:
    def test_build_vocab(self):
        texts = ["hello 😠", "world 🔥", "test 😤"]
        vocab = build_vocab_from_texts(texts)
        assert "😠" in vocab
        assert "🔥" in vocab
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1

    def test_unk_fallback(self):
        vocab = build_vocab_from_texts(["😠"])
        idx = get_emoji_index("🛸", vocab)   # unseen emoji
        assert idx == vocab["<UNK>"]

    def test_known_emoji_lookup(self):
        vocab = build_vocab_from_texts(["😠"])
        idx = get_emoji_index("😠", vocab)
        assert idx >= 2   # 0=PAD, 1=UNK, 2+=emoji


class TestNLPPipeline:
    @pytest.fixture(scope="class")
    def nlp(self):
        return build_emoji_aware_nlp()

    def test_tokenize_returns_list(self, nlp):
        result = tokenize("order #782 not delivered 😠", nlp)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_emoji_is_separate_token(self, nlp):
        result = tokenize("notdelivered😠", nlp)
        assert "😠" in result

    def test_empty_string(self, nlp):
        result = tokenize("", nlp)
        assert isinstance(result, list)
