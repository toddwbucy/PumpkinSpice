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

Norm handling (an operationalization choice, deliberately explicit). ``transformers``
exposes the final residual only AFTER the model's final norm (RMSNorm/LayerNorm) as
``output_hidden_states[-1]`` -- the true residual leaving the last block is never in
that tuple. Using it would distort the last layer's block update and rescale every
trajectory point. So this driver ignores ``output_hidden_states`` and reads the TRUE
pre-norm residual stream directly with hooks: a forward hook on each block (its output
= the residual leaving it) and a forward-pre-hook on layer 0 (its input = the token
embeddings actually fed in, including any Gemma-style sqrt(d) scaling). Every per-layer
rho is then on one pre-norm footing, and the #7 trajectory is the pre-final-norm final
residual. The post-final-norm stream is the alternative reading, not used here.

Hooks needed: one per block (block output) + one per block's MLP (isolated MLP update)
+ one pre-hook on layer 0 (embeddings). rho_block and the #7 trajectory then come from
the block outputs; the MLP's own input residual is recovered as ``h_out - delta_mlp``,
so the attention sublayer is never hooked.

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
    forward pass; raw residuals are never returned).

    ``frozen`` gives shallow immutability only -- ``d_rho`` and the arrays remain
    mutable, so do not rely on this for hashing or caching.
    """

    d_rho: dict[float, int]  # variance fraction -> effective dimension (#7)
    kinematics: EarlyKinematics  # early kinematics of the trajectory (#7)
    rho_block: Array  # (n_layers,) full-block novelty per layer (#8)
    rho_mlp: Array  # (n_layers,) MLP-alone novelty per layer (#8)
    n_prompt_tokens: int
    n_output_tokens: int
    n_layers: int
    # Load dtype the forward ran at (bf16 vs fp32 perturb d_rho/rho ~0.3%). Recorded
    # as provenance so a floor-test corpus cannot silently pool incommensurable
    # precisions; the evaluator rejects a mix. Default marks pre-provenance metrics.
    dtype: str = "unknown"


def _find_decoder_layers(model: Any) -> Any:
    """Locate the list of transformer blocks across common architectures.

    Covers the Llama-family layout (Qwen / Mistral / Ministral / Gemma:
    ``model.model.layers``), GPT-2 (``model.transformer.h``, used by the tiny test
    model), and GPT-NeoX (``model.gpt_neox.layers``). Raises with a clear message if
    none is present.
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
        "(Llama-family), model.transformer.h (GPT-2), and model.gpt_neox.layers (GPT-NeoX)"
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
        # Provenance: the parameter dtype the forward runs at (e.g. "bfloat16").
        self.dtype = str(getattr(model, "dtype", "unknown")).replace("torch.", "")
        # The base decoder (no lm_head) is what we forward through -- get_decoder() is
        # the transformers-standard accessor; fall back to the model itself (the tiny
        # test model exposes get_decoder too).
        self._base = model.get_decoder() if hasattr(model, "get_decoder") else model
        layers = _find_decoder_layers(model)
        self.n_layers = len(layers)
        # Per-forward caches, cleared at the start of every replay: the residual
        # entering layer 0 (embeddings), each block's output, each MLP's output.
        self._block_in: Any = None
        self._block_out: dict[int, Any] = {}
        self._mlp_out: dict[int, Any] = {}
        self._handles: list[Any] = []
        # Pre-hook on layer 0 captures the true embeddings fed to the stack (post any
        # architecture-specific scaling), which output_hidden_states[0] would also give
        # but only alongside the full -- and post-final-norm -- tuple.
        self._handles.append(
            layers[0].register_forward_pre_hook(self._block_in_hook, with_kwargs=True)
        )
        for idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                self.close()
                raise ValueError(f"layer {idx} has no `.mlp` submodule; cannot capture rho_MLP")
            self._handles.append(layer.register_forward_hook(self._make_block_hook(idx)))
            self._handles.append(mlp.register_forward_hook(self._make_mlp_hook(idx)))

    @staticmethod
    def _offload(tensor: Any) -> Any:
        # Copy the captured tensor to host RAM immediately. Otherwise every layer's
        # block + MLP output would accumulate on the GPU for the whole forward, and a
        # long trajectory (36 layers x thousands of tokens x hidden) OOMs even a 48GB
        # card. The copy leaves the original on-device for the rest of the forward.
        return None if tensor is None else tensor.detach().to("cpu")

    def _block_in_hook(self, _module: Any, args: Any, kwargs: Any) -> None:
        # Decoder layers receive hidden_states as the first positional arg (or, rarely,
        # as a kwarg); handle both so this does not depend on the call convention.
        self._block_in = self._offload(args[0] if args else kwargs.get("hidden_states"))

    def _make_block_hook(self, idx: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._block_out[idx] = self._offload(output[0] if isinstance(output, tuple) else output)

        return hook

    def _make_mlp_hook(self, idx: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            # Some MLP modules return a tuple; the update tensor is the first element.
            self._mlp_out[idx] = self._offload(output[0] if isinstance(output, tuple) else output)

        return hook

    def close(self) -> None:
        """Remove all forward hooks. Safe to call more than once."""
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
        load_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ReplayModel:
        """Load a causal LM + tokenizer from the Hub or a local path.

        Pass ``gguf_file`` to dequantize the SAME GGUF the harness served (matching
        the numerical object that produced the trace); support is limited to the
        architectures transformers can convert. Extra ``from_pretrained`` loading
        options (e.g. ``device_map``, ``low_cpu_mem_usage`` for large models, which
        also want ``accelerate`` installed) go through ``load_kwargs``; remaining
        ``**kwargs`` configure the ReplayModel (the operationalization knobs).
        """
        try:
            import torch
            import transformers
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the 'replay' extra is required for ReplayModel.from_pretrained "
                "(uv sync --extra replay)"
            ) from exc

        # Default to SDPA attention: eager attention materializes the full seq x seq
        # score matrix, which OOMs on long reasoning trajectories (a 10k-token trace
        # needs a ~15GB single allocation). SDPA is O(seq) memory and gives the same
        # residual stream we read, so the geometry is unaffected. Overridable via
        # load_kwargs (e.g. attn_implementation="flash_attention_2").
        load_kw: dict[str, Any] = {
            "torch_dtype": getattr(torch, dtype),
            "attn_implementation": "sdpa",
            **(load_kwargs or {}),
        }
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
        continuation concatenates at a known boundary.

        Note: ``output`` is re-tokenized independently, so at the prompt/output seam
        the token ids can differ slightly (leading-space / BPE-merge effects) from
        what the model originally emitted. Harmless for the geometry, but it means the
        replayed ids are not guaranteed byte-identical to the generation.
        """
        import torch

        if self.chat_template and getattr(self.tokenizer, "chat_template", None):
            # Reproduce what the model actually processed: the chat wrapper the decoder's
            # /v1/chat/completions server applied around the user content. Render the
            # template to a STRING then tokenize it (add_special_tokens=False -- the
            # template already emits BOS/role markers). This is what the vLLM server does
            # internally, and unlike apply_chat_template(tokenize=True) -- which in recent
            # transformers returns a BatchEncoding of Encoding objects, not a flat id list
            # -- it yields a plain list[int] across versions.
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
            )
            prompt_ids = self.tokenizer(str(text), add_special_tokens=False)["input_ids"]
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
        the geometry functionals over the chosen span. ``input_ids`` must be a single
        ``(1, S)`` sequence."""
        import torch

        input_ids = input_ids.to(self.model.device)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"expected a single (1, S) sequence; got shape {tuple(input_ids.shape)}"
            )
        seq_len = int(input_ids.shape[1])
        if not 0 <= n_prompt_tokens <= seq_len:
            raise ValueError(f"n_prompt_tokens must be in [0, {seq_len}]; got {n_prompt_tokens}")

        self._block_in = None
        self._block_out.clear()
        self._mlp_out.clear()
        with torch.no_grad():
            # Run the base decoder, NOT the full CausalLM: the lm_head projects the whole
            # sequence to [seq, vocab] logits (~15GB for a 10k-token trace over a 150k
            # vocab) that we never use -- we read the residual stream via hooks. The base
            # model still runs every layer, so all hooks fire.
            self._base(input_ids=input_ids, use_cache=False)
        if (
            self._block_in is None
            or len(self._block_out) != self.n_layers
            or len(self._mlp_out) != self.n_layers
        ):
            raise RuntimeError(
                f"instrumentation incomplete: block_in={'set' if self._block_in is not None else 'missing'}, "
                f"{len(self._block_out)} block / {len(self._mlp_out)} mlp outputs for {self.n_layers} layers"
            )

        n_output_tokens = seq_len - n_prompt_tokens
        lo = n_prompt_tokens if self.trajectory_span == "output" else 0
        if seq_len - lo < 2:
            raise ValueError(
                f"trajectory span has {seq_len - lo} token(s); need >= 2 for velocity/covariance "
                f"(span={self.trajectory_span!r}, prompt={n_prompt_tokens}, seq={seq_len})"
            )

        block_out = {i: _to_numpy(self._block_out[i]) for i in range(self.n_layers)}
        embeddings = _to_numpy(self._block_in)

        # #7 trajectory: the TRUE final block output (pre-final-norm residual). See class docs.
        final = block_out[self.n_layers - 1][lo:]
        d_rho = {rho: effective_dimension(final, rho) for rho in self.rho_thresholds}
        kinematics = early_kinematics(final, self.kinematics_fraction)

        rho_block = np.empty(self.n_layers, dtype=np.float64)
        rho_mlp = np.empty(self.n_layers, dtype=np.float64)
        prev = embeddings  # residual entering layer 0
        for i in range(self.n_layers):
            cur = block_out[i]  # residual leaving layer i (pre-norm, from the hook)
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
            dtype=self.dtype,
        )
