"""Teacher-forced replay driver: recorded turn -> hidden-state trajectory metrics
(issues #7, #8).

This is the analysis-side extractor. Given a recorded ``(prompt, output)`` it forces
the output as the continuation of the prompt through one instrumented forward pass,
reads the residual stream, and reduces it to the pure functionals in
``pumpkinspice.introspect.geometry``. It never touches the live decoder or the
runtime loop -- it re-derives trajectories offline from full-payload captures, at
zero cost to any live forward pass.

Signals from ONE forward pass:
  * #7 -- the final-layer trajectory (d_rho + early kinematics).
  * #8 -- per-layer novelty rho_block / rho_MLP.

``output_hidden_states=True`` already yields the residual entering every block and
the full-block update (a difference of consecutive hidden states), so rho_block and
the #7 trajectory need no hooks. Only rho_MLP needs the MLP submodule's isolated
output, captured by one forward hook per layer; the MLP's own input residual is then
recovered for free as ``h_out - delta_mlp`` (no attention hook needed).

The operationalization knobs that geometry.py deliberately left to the call site are
constructor params here, with documented defaults flagged for pre-registration:
  * ``rho_thresholds``  -- variance fractions for d_rho (default 0.5/0.75/0.9).
  * ``kinematics_fraction`` -- the "first fifth" window (default 0.2).
  * ``trajectory_span`` -- which token positions form the trajectory / rho average:
    "output" (the forced continuation, default) or "full" (prompt + output).
  * ``mlp_residual`` -- which residual rho_MLP is measured against: "block_in"
    (the residual entering the whole block, #8's literal phrasing, default) or
    "mlp_in" (the residual entering the MLP sublayer, h_out - delta_mlp).

Heavy deps (torch, transformers) live behind the ``replay`` extra and are imported
lazily, so importing this module -- or using geometry.py -- costs nothing without them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from pumpkinspice.introspect.geometry import (
    Array,
    EarlyKinematics,
    early_kinematics,
    effective_dimension,
    mean_token_cosine,
)

TrajectorySpan = Literal["output", "full"]
MlpResidual = Literal["block_in", "mlp_in"]


@dataclass(frozen=True)
class TrajectoryMetrics:
    """Per-turn trajectory-geometry metrics (the compact reduction of one replayed
    forward pass; raw residuals are never returned)."""

    d_rho: dict[float, int]  # variance fraction -> effective dimension (#7)
    kinematics: EarlyKinematics  # early kinematics of the trajectory (#7)
    rho_block: Array  # (n_layers,) full-block novelty per layer (#8)
    rho_mlp: Array  # (n_layers,) MLP-alone novelty per layer (#8)
    n_prompt_tokens: int
    n_output_tokens: int
    n_layers: int


def _find_decoder_layers(model: Any) -> Any:
    """Locate the list of transformer blocks across common architectures.

    Covers the Llama-family layout (Qwen / Mistral / Ministral / Gemma:
    ``model.model.layers``) and GPT-2 (``model.transformer.h``, used by the tiny
    test model). Raises with a clear message if neither is present.
    """
    for path in (("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise ValueError(
        "could not locate decoder layers; supported layouts are model.model.layers "
        "(Llama-family) and model.transformer.h (GPT-2)"
    )


def _to_numpy(tensor: Any) -> Array:
    """First batch row of a hidden-state tensor -> (T, d) float64 numpy.

    Uses the ``.float()`` tensor method (not ``torch.float32``) so this needs no
    torch symbol at import time. ``np.asarray`` launders the tensor's ``Any`` back
    to a typed array for mypy.
    """
    return np.asarray(tensor[0].detach().float().cpu().numpy(), dtype=np.float64)


class ReplayModel:
    """Replays recorded turns through an instrumented forward pass.

    Load once (expensive), replay many. The model and tokenizer are injected so the
    extraction path is testable on a tiny from-config model; use ``from_pretrained``
    for the normal case.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        rho_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9),
        kinematics_fraction: float = 0.2,
        trajectory_span: TrajectorySpan = "output",
        mlp_residual: MlpResidual = "block_in",
        chat_template: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.rho_thresholds = tuple(rho_thresholds)
        self.kinematics_fraction = kinematics_fraction
        self.trajectory_span = trajectory_span
        self.mlp_residual = mlp_residual
        self.chat_template = chat_template

        model.eval()
        layers = _find_decoder_layers(model)
        self.n_layers = len(layers)
        # One forward hook per layer captures the MLP's isolated output into a cache
        # keyed by layer index; the cache is cleared at the start of every replay.
        self._mlp_out: dict[int, Any] = {}
        self._handles: list[Any] = []
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                self.close()
                raise ValueError(f"layer {idx} has no `.mlp` submodule; cannot capture rho_MLP")
            self._handles.append(mlp.register_forward_hook(self._make_mlp_hook(idx)))

    def _make_mlp_hook(self, idx: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # Some MLP modules return a tuple; the update tensor is the first element.
            self._mlp_out[idx] = output[0] if isinstance(output, tuple) else output

        return hook

    def close(self) -> None:
        """Remove the forward hooks. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> ReplayModel:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        gguf_file: str | None = None,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs: Any,
    ) -> ReplayModel:
        """Load a causal LM + tokenizer from the Hub or a local path.

        Pass ``gguf_file`` to dequantize the SAME GGUF the harness served (matching
        the numerical object that produced the trace); support is limited to the
        architectures transformers can convert. Large models may need ``accelerate``
        and a ``device`` / ``device_map`` the caller manages.
        """
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the 'replay' extra is required for ReplayModel.from_pretrained "
                "(uv sync --extra replay)"
            ) from exc

        load_kw: dict[str, Any] = {"torch_dtype": getattr(torch, dtype)}
        tok_kw: dict[str, Any] = {}
        if gguf_file is not None:
            load_kw["gguf_file"] = gguf_file
            tok_kw["gguf_file"] = gguf_file
        # Kept Any: transformers ships partial types whose .to() overloads misfire on
        # a str device; the driver treats these objects as untyped by design.
        model: Any = transformers.AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
        model.to(device)
        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(model_id, **tok_kw)
        return cls(model, tokenizer, **kwargs)

    def _encode(self, prompt: str, output: str) -> tuple[Any, int]:
        """Build teacher-forced input_ids = prompt tokens + output tokens, and the
        prompt length. The output is appended without special tokens so the forced
        continuation concatenates cleanly at a known boundary."""
        import torch

        if self.chat_template and getattr(self.tokenizer, "chat_template", None):
            # Reproduce what the model actually processed: the chat wrapper the decoder's
            # /v1/chat/completions server applied around the user content.
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True
            )
        else:
            prompt_ids = self.tokenizer(prompt, add_special_tokens=True)["input_ids"]
        output_ids = self.tokenizer(output, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([list(prompt_ids) + list(output_ids)], device=self.model.device)
        return input_ids, len(prompt_ids)

    def replay(self, prompt: str, output: str) -> TrajectoryMetrics:
        """Replay one recorded turn. ``output`` is the full generated text to force
        (for a reasoning model, the caller assembles reasoning + answer)."""
        input_ids, n_prompt_tokens = self._encode(prompt, output)
        return self.replay_token_ids(input_ids, n_prompt_tokens)

    def replay_token_ids(self, input_ids: Any, n_prompt_tokens: int) -> TrajectoryMetrics:
        """Core extraction on pre-tokenized ids: one instrumented forward pass ->
        the geometry functionals over the chosen span."""
        import torch

        input_ids = input_ids.to(self.model.device)
        self._mlp_out.clear()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        hidden = out.hidden_states  # tuple of (n_layers + 1) tensors, each (1, S, d)
        if len(self._mlp_out) != self.n_layers:
            raise RuntimeError(
                f"captured {len(self._mlp_out)} MLP outputs but model has {self.n_layers} layers"
            )

        seq_len = int(input_ids.shape[1])
        n_output_tokens = seq_len - n_prompt_tokens
        lo = n_prompt_tokens if self.trajectory_span == "output" else 0
        if seq_len - lo < 2:
            raise ValueError(
                f"trajectory span has {seq_len - lo} token(s); need >= 2 for velocity/covariance "
                f"(span={self.trajectory_span!r}, prompt={n_prompt_tokens}, seq={seq_len})"
            )

        final = _to_numpy(hidden[-1])[lo:]
        d_rho = {rho: effective_dimension(final, rho) for rho in self.rho_thresholds}
        kinematics = early_kinematics(final, self.kinematics_fraction)

        rho_block = np.empty(self.n_layers, dtype=np.float64)
        rho_mlp = np.empty(self.n_layers, dtype=np.float64)
        prev = _to_numpy(hidden[0])  # residual entering layer 0 (embeddings)
        for i in range(self.n_layers):
            cur = _to_numpy(hidden[i + 1])  # residual leaving layer i
            h_in, h_out = prev[lo:], cur[lo:]
            delta_block = h_out - h_in
            delta_mlp = _to_numpy(self._mlp_out[i])[lo:]
            # "block_in": novelty vs the residual entering the whole block (#8's literal
            # phrasing). "mlp_in": vs the residual entering the MLP sublayer, h_out - delta_mlp.
            residual = h_in if self.mlp_residual == "block_in" else (h_out - delta_mlp)
            rho_block[i] = mean_token_cosine(delta_block, h_in)
            rho_mlp[i] = mean_token_cosine(delta_mlp, residual)
            prev = cur

        return TrajectoryMetrics(
            d_rho=d_rho,
            kinematics=kinematics,
            rho_block=rho_block,
            rho_mlp=rho_mlp,
            n_prompt_tokens=n_prompt_tokens,
            n_output_tokens=n_output_tokens,
            n_layers=self.n_layers,
        )
