"""Smoke/integration tests for the teacher-forced replay driver (issues #7, #8).

No network and no download: a 2-layer random-weight Llama built from config
exercises the exact ``model.model.layers[i].mlp`` path the real Qwen/Mistral models
use. These tests verify the driver WIRES the forward pass to the metrics correctly
(shapes, spans, knobs, hook lifecycle); the metric MATH itself is proven in
test_geometry.py, and scientific validity is the operator's real-model job.

Skips cleanly unless the ``replay`` extra (torch + transformers) is installed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pumpkinspice.introspect.replay import (  # noqa: E402  (after importorskip)
    ReplayModel,
    _find_decoder_layers,
)


def _tiny_llama() -> object:
    torch.manual_seed(0)  # deterministic random weights, no flake risk
    cfg = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return transformers.LlamaForCausalLM(cfg)


def _ids(seq_len: int) -> object:
    # Deterministic, in-vocab token ids as a single (1, S) sequence.
    return torch.arange(seq_len, dtype=torch.long).remainder(32).unsqueeze(0)


def test_replay_produces_wellformed_metrics() -> None:
    rm = ReplayModel(_tiny_llama(), tokenizer=None)
    m = rm.replay_token_ids(_ids(9), n_prompt_tokens=3)

    assert m.n_prompt_tokens == 3
    assert m.n_output_tokens == 6
    assert m.n_layers == 2

    # d_rho: one entry per default threshold, each a count within the span length.
    assert set(m.d_rho) == {0.5, 0.75, 0.9}
    assert all(0 <= v <= 6 for v in m.d_rho.values())
    assert m.d_rho[0.5] <= m.d_rho[0.9]  # monotone in rho

    assert m.kinematics.n_points >= 2

    # per-layer novelty curves: one value per layer, finite, bounded like a cosine.
    for curve in (m.rho_block, m.rho_mlp):
        assert curve.shape == (2,)
        assert bool(torch.isfinite(torch.from_numpy(curve)).all())
        assert curve.min() >= -1.0 - 1e-9 and curve.max() <= 1.0 + 1e-9
    rm.close()


def test_trajectory_span_full_uses_all_tokens() -> None:
    rm = ReplayModel(_tiny_llama(), tokenizer=None, trajectory_span="full")
    m = rm.replay_token_ids(_ids(9), n_prompt_tokens=3)
    # "full" spans all 9 positions, so d_rho may reach beyond the 6 output tokens.
    assert all(0 <= v <= 9 for v in m.d_rho.values())
    rm.close()


def test_mlp_residual_variants_both_run() -> None:
    ids = _ids(9)
    block_in = ReplayModel(_tiny_llama(), tokenizer=None, mlp_residual="block_in")
    mlp_in = ReplayModel(_tiny_llama(), tokenizer=None, mlp_residual="mlp_in")
    a = block_in.replay_token_ids(ids, 3)
    b = mlp_in.replay_token_ids(ids, 3)
    assert a.rho_mlp.shape == b.rho_mlp.shape == (2,)
    block_in.close()
    mlp_in.close()


def test_short_span_raises() -> None:
    rm = ReplayModel(_tiny_llama(), tokenizer=None)
    with pytest.raises(ValueError, match="need >= 2"):
        rm.replay_token_ids(_ids(9), n_prompt_tokens=8)  # output span = 1
    rm.close()


def test_context_manager_removes_hooks() -> None:
    with ReplayModel(_tiny_llama(), tokenizer=None) as rm:
        assert len(rm._handles) == 5  # 1 pre-hook + (block + mlp) per 2 layers
    assert rm._handles == []
    # With hooks gone, the caches no longer fill -> the guard fires.
    with pytest.raises(RuntimeError, match="instrumentation incomplete"):
        rm.replay_token_ids(_ids(9), n_prompt_tokens=3)


def test_input_validation() -> None:
    rm = ReplayModel(_tiny_llama(), tokenizer=None)
    with pytest.raises(ValueError, match=r"single \(1, S\) sequence"):
        rm.replay_token_ids(torch.zeros((2, 6), dtype=torch.long), n_prompt_tokens=2)
    with pytest.raises(ValueError, match="n_prompt_tokens must be in"):
        rm.replay_token_ids(_ids(9), n_prompt_tokens=99)  # > seq_len
    rm.close()


def test_find_decoder_layers_llama_and_unknown() -> None:
    model = _tiny_llama()
    assert len(_find_decoder_layers(model)) == 2

    class _Bare:
        pass

    with pytest.raises(ValueError, match="could not locate decoder layers"):
        _find_decoder_layers(_Bare())


def test_missing_mlp_submodule_raises_and_cleans_up() -> None:
    model = _tiny_llama()
    delattr(model.model.layers[1], "mlp")  # second block loses its MLP
    with pytest.raises(ValueError, match=r"no `.mlp`"):
        ReplayModel(model, tokenizer=None)


class _FakeTok:
    """Minimal tokenizer: 1 token per character, ids in-vocab (1..31).

    Models a REAL tokenizer's chat path: apply_chat_template(tokenize=False) renders
    to a STRING (here a 2-char '>' + content marker), which the driver then tokenizes
    -- NOT the id list a too-kind fake would return directly (that masked the
    BatchEncoding/string bug that only a real tokenizer exposes)."""

    def __init__(self, chat_template: str | None = None) -> None:
        self.chat_template = chat_template

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [(ord(c) % 31) + 1 for c in text]}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str:
        return ">>" + messages[0]["content"]  # rendered template string, not ids


def test_replay_plain_tokenizer_path() -> None:
    rm = ReplayModel(_tiny_llama(), tokenizer=_FakeTok(chat_template=None))
    m = rm.replay("hello world", "abcdef")  # 11 prompt chars, 6 output chars
    assert m.n_prompt_tokens == 11
    assert m.n_output_tokens == 6
    rm.close()


def test_replay_chat_template_path() -> None:
    # A non-empty chat_template routes _encode through apply_chat_template, which
    # renders to a string that is then tokenized: ">>hey" -> 5 prompt tokens. (A
    # regression here catches the apply_chat_template-returns-a-string bug.)
    rm = ReplayModel(_tiny_llama(), tokenizer=_FakeTok(chat_template="tmpl"))
    m = rm.replay("hey", "abcdef")
    assert m.n_prompt_tokens == 5  # 2-char ">>" marker + 3-char content, tokenized
    assert m.n_output_tokens == 6
    rm.close()
