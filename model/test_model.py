"""Tests for Layer 2: BiLSTM-CRF Model — shape correctness and output validity."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from layers.layer1_preprocessing.emoji_vocab import build_vocab_from_texts
from layers.layer2_bilstm_crf.model import BiLSTMCRF
from layers.layer2_bilstm_crf.crf import CRFLayer
from config import NUM_TAGS, EMBEDDING_DIM


@pytest.fixture(scope="module")
def emoji_vocab():
    return build_vocab_from_texts(["😠", "😤", "💀", "🔥"])


@pytest.fixture(scope="module")
def model(emoji_vocab):
    return BiLSTMCRF(emoji_vocab=emoji_vocab)


class TestCRFLayer:
    def test_neg_log_likelihood_shape(self):
        crf = CRFLayer(num_tags=NUM_TAGS)
        B, L = 2, 8
        emissions = torch.randn(B, L, NUM_TAGS)
        tags  = torch.zeros(B, L, dtype=torch.long)
        mask  = torch.ones(B, L, dtype=torch.bool)
        loss  = crf.neg_log_likelihood(emissions, tags, mask)
        assert loss.shape == torch.Size([])   # scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_viterbi_output_length(self):
        crf = CRFLayer(num_tags=NUM_TAGS)
        B, L = 2, 6
        emissions = torch.randn(B, L, NUM_TAGS)
        mask = torch.ones(B, L, dtype=torch.bool)
        seqs = crf.viterbi(emissions, mask)
        assert len(seqs) == B
        for s in seqs:
            assert len(s) == L

    def test_viterbi_valid_tag_indices(self):
        crf = CRFLayer(num_tags=NUM_TAGS)
        B, L = 1, 10
        emissions = torch.randn(B, L, NUM_TAGS)
        mask = torch.ones(B, L, dtype=torch.bool)
        seqs = crf.viterbi(emissions, mask)
        for tag in seqs[0]:
            assert 0 <= tag < NUM_TAGS


class TestBiLSTMCRF:
    def test_forward_returns_scalar_loss(self, model, emoji_vocab):
        B, L = 2, 7
        spacy_v = torch.randn(B, L, 300)
        emoji   = torch.zeros(B, L, dtype=torch.long)
        tags    = torch.zeros(B, L, dtype=torch.long)
        mask    = torch.ones(B, L, dtype=torch.bool)
        loss = model(spacy_v, emoji, tags, mask)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)

    def test_decode_returns_list(self, model):
        B, L = 1, 5
        spacy_v = torch.randn(B, L, 300)
        emoji   = torch.zeros(B, L, dtype=torch.long)
        mask    = torch.ones(B, L, dtype=torch.bool)
        result = model.decode(spacy_v, emoji, mask)
        assert isinstance(result, list)
        assert isinstance(result[0], list)
        assert len(result[0]) == L

    def test_embedding_output_dim(self, model):
        """Embedding layer should produce (1, L, EMBEDDING_DIM) tensors."""
        B, L = 1, 4
        spacy_v = torch.randn(B, L, 300)
        emoji   = torch.zeros(B, L, dtype=torch.long)
        embedded = model.embedding(spacy_v, emoji)
        assert embedded.shape == (B, L, EMBEDDING_DIM)

    def test_variable_length_sequences(self, model):
        """Model should handle varying sequence lengths in same batch."""
        spacy_v = torch.randn(2, 10, 300)
        emoji   = torch.zeros(2, 10, dtype=torch.long)
        mask    = torch.tensor([[True]*8 + [False]*2,
                                [True]*5 + [False]*5])
        result = model.decode(spacy_v, emoji, mask)
        assert len(result) == 2
