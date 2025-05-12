# Copyright © 2024-2025
#
# Continuous batching scheduler for mlx-lm.
#
# Usage:
#     from mlx_lm.continuous_batcher import ContinuousBatcher
#
#     batcher = ContinuousBatcher(model, tokenizer, sampler=my_sampler)
#     req1 = batcher.submit("Hello world!", max_tokens=32)
#     req2 = batcher.submit("What is the capital of France?", max_tokens=16)
#     while not batcher.empty:
#         finished = batcher.step()          # one forward call for ALL live reqs
#         for rsp in finished:               # list[AsyncResponse]
#             if rsp.finished:
#                 print(rsp.text)

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .models import cache
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper
from .generate import maybe_quantize_kv_cache


# -------------------------------------------------------------------------
# Public dataclasses -------------------------------------------------------
# -------------------------------------------------------------------------

@dataclass
class AsyncResponse:
    request_id: int
    token: int
    logprobs: mx.array
    text_segment: str
    finished: bool
    finish_reason: Optional[str] = None


@dataclass
class _RequestState:
    # internal, not exported
    prompt_tokens: mx.array
    detokenizer: Any
    cache_slices: List[slice]                 # one slice per layer in global cache
    max_tokens: int
    tokens_generated: int = 0
    finished: bool = False
    finish_reason: Optional[str] = None
    quantize_after: int = 0                   # per-request threshold


# -------------------------------------------------------------------------
# Helper utils ------------------------------------------------------------
# -------------------------------------------------------------------------

def _broadcast(fn: Callable[[mx.array], mx.array]) -> Callable[[mx.array], mx.array]:
    """
    Wrap a (V,) -> int sampler so it can accept (B,V) logits.
    """
    def _wrapped(logits: mx.array) -> mx.array:
        shape = logits.shape
        logits = logits.reshape(-1, shape[-1])
        out = fn(logits)
        return out.reshape(shape[:-1])
    return _wrapped


def _concatenate_cache(dest: List[Any], src: List[Any]) -> List[slice]:
    """
    Append `src` (a per-layer KV cache list) onto `dest` in-place.
    Returns the slice list that selects *just* the newly appended rows.
    """
    slices = []
    for layer_idx, (d, s) in enumerate(zip(dest, src)):
        start = d.keys.shape[-2]             # K shape (..., L, Hd)
        # copy keys/values
        d.keys = mx.concatenate([d.keys, s.keys], axis=-2)
        d.values = mx.concatenate([d.values, s.values], axis=-2)
        d.offset = d.keys.shape[-2]
        end = d.keys.shape[-2]
        slices.append(slice(start, end))
    return slices


def _slice_cache(global_cache: List[Any], slices: List[slice]) -> List[Any]:
    """
    Return a *view* of global_cache restricted to `slices`.
    Each element is still a real KVCache, but points into the same mx.arrays.
    """
    view = []
    for g, slc in zip(global_cache, slices):
        view.append(g.view(slc))
    return view


# -------------------------------------------------------------------------
# Main class --------------------------------------------------------------
# -------------------------------------------------------------------------

class ContinuousBatcher:
    """
    High-level scheduler that batches arbitrary numbers of simultaneous
    generation requests into one forward call per timestep.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        *,
        sampler: Callable[[mx.array], mx.array] | None = None,
        logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
        quantized_kv_start: int = 5000,
        temp: float = 0.0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.requests: dict[int, _RequestState] = {}
        self._next_id = itertools.count().__next__

        # Global batched KV cache (list per layer)
        self.global_cache = cache.make_prompt_cache(
            model, max_kv_size=max_kv_size
        )

        # Sampler – default is identical to generate.py’s logic
        if sampler is None:
            sampler = make_sampler(
                temp,
                top_p,
                min_p,
                min_tokens_to_keep,
                top_k=top_k,
                xtc_probability=xtc_probability,
                xtc_threshold=xtc_threshold,
                xtc_special_tokens=tokenizer.encode("\n") + list(tokenizer.eos_token_ids),
            )
        self.sampler = _broadcast(sampler)
        self.logits_processors = logits_processors or []

        # KV-quantisation settings
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start

        # book-keeping
        self.empty = True
        self._stream = mx.new_stream(mx.default_device())

    # ------------------------------------------------------------------
    # Public API -------------------------------------------------------
    # ------------------------------------------------------------------

    def submit(
        self,
        prompt: Union[str, List[int], mx.array],
        *,
        max_tokens: int = 256,
        quantize_after: Optional[int] = None,
    ) -> int:
        """
        Register a new generation job and return its internal request_id.
        """
        if not isinstance(prompt, mx.array):
            prompt = self.tokenizer.encode(prompt) if isinstance(prompt, str) else mx.array(prompt)
        prompt = prompt.astype(mx.uint32)

        req_cache = cache.make_prompt_cache(
            self.model, max_kv_size=None
        )

        # Prefill prompt (we do it request-by-request; could batch optimise later)
        with mx.stream(self._stream):
            self.model(prompt[None], cache=req_cache)
            mx.eval([c.state for c in req_cache])

        # Add to global cache
        cache_slices = _concatenate_cache(self.global_cache, req_cache)

        detok = self.tokenizer.detokenizer
        detok.reset()

        rid = self._next_id()
        self.requests[rid] = _RequestState(
            prompt_tokens=prompt,
            detokenizer=detok,
            cache_slices=cache_slices,
            max_tokens=max_tokens,
            quantize_after=quantize_after or self.quantized_kv_start,
        )
        self.empty = False
        return rid

    def step(self) -> List[AsyncResponse]:
        """
        Execute **one** forward pass for *all live* requests.
        Returns a list of AsyncResponse objects (could be empty).
        """
        if not self.requests:
            self.empty = True
            return []

        live_ids = [rid for rid, r in self.requests.items() if not r.finished]
        if not live_ids:
            self.empty = True
            return []

        # 1. Prepare batched input token tensor --------------------------------
        tok_tensor = mx.array(
            [self.requests[rid].prompt_tokens[-1] for rid in live_ids],
            dtype=mx.uint32,
        )[:, None]  # (B,1)

        # 2. Slice out per-request KV caches and concatenate -------------------
        batch_cache = []
        for layer_idx in range(len(self.global_cache)):
            layer_views = [
                self.global_cache[layer_idx].view(self.requests[rid].cache_slices[layer_idx])
                for rid in live_ids
            ]
            batch_cache.append(cache.concat(layer_views))   # helper inside cache.py

        # 3. Forward -----------------------------------------------------------
        with mx.stream(self._stream):
            logits = self.model(tok_tensor, cache=batch_cache)[:, -1, :]  # (B,V)

        # 4. Apply processors & sampler ---------------------------------------
        for proc in self.logits_processors:
            logits = proc(None, logits)      # broadcast processors handle (B,V)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        next_tokens = self.sampler(logprobs)            # (B,)

        # 5. Post-process each request ----------------------------------------
        responses: List[AsyncResponse] = []
        for i, rid in enumerate(live_ids):
            req = self.requests[rid]
            token_i = int(next_tokens[i].item())
            req.detokenizer.add_token(token_i)
            text_seg = req.detokenizer.last_segment

            req.tokens_generated += 1
            finished = (
                token_i in self.tokenizer.eos_token_ids or
                req.tokens_generated >= req.max_tokens
            )
            req.finished = finished
            if finished:
                req.detokenizer.finalize()
                req.finish_reason = (
                    "stop" if token_i in self.tokenizer.eos_token_ids else "length"
                )

            # Quantise this request’s cache if threshold crossed
            maybe_quantize_kv_cache(
                [_slice_cache(self.global_cache, req.cache_slices)[0]],
                req.quantize_after,
                self.kv_group_size,
                self.kv_bits,
            )

            responses.append(
                AsyncResponse(
                    request_id=rid,
                    token=token_i,
                    logprobs=logprobs[i],
                    text_segment=text_seg,
                    finished=finished,
                    finish_reason=req.finish_reason,
                )
            )

        # 6. Clean up finished requests (and free KV slices) -------------------
        for rsp in responses:
            if rsp.finished:
                self._remove_request(rsp.request_id)

        return responses

    # ------------------------------------------------------------------
    # Internal helpers -------------------------------------------------
    # ------------------------------------------------------------------

    def _remove_request(self, rid: int):
        """
        Physically remove KV slices for a finished request and compact the
        global cache in-place.
        """
        req = self.requests.pop(rid, None)
        if req is None:
            return
        for layer_idx, slc in enumerate(req.cache_slices):
            cache.trim_prompt_cache_slice(self.global_cache[layer_idx], slc)
        if not self.requests:
            self.empty = True


# mlx_lm/models/batched_rope.py
# Copyright © 2023-2024 Apple Inc. (and modifications for batching)

import mlx.core as mx
import mlx.nn as nn
from typing import Union

class BatchedRoPEUtility(nn.Module):
    """
    Applies Rotary Positional Embeddings to an input tensor using
    pre-computed frequencies and a batch of per-sequence offsets.
    This utility does not compute frequencies itself.
    """
    def __init__(self, dims: int):
        super().__init__()
        self.dims = dims # The number of dimensions to rotate (typically head_dim)

    def __call__(
        self,
        x: mx.array,
        computed_frequencies: mx.array,
        offsets: mx.array,
    ) -> mx.array:
        """
        Apply RoPE with batched offsets.

        Args:
            x (mx.array): Input tensor. Expected shape (B, NumHeads, SeqLen, HeadDims).
                          Can also handle (B, SeqLen, HeadDims) or (SeqLen, HeadDims)
                          which will be reshaped internally.
            computed_frequencies (mx.array): The pre-computed inverse frequencies (theta_i).
                                           Expected shape like (1, Dims//2) or broadcastable
                                           to (1, 1, 1, Dims//2) for `angles` calculation.
                                           Typically, this is `base ** (arange(0, dims, 2)/dims)`
                                           or a variant from specialized RoPE classes.
            offsets (mx.array): A 1D array of per-sequence offsets. Shape (B,).
                                Each offset is the starting position for the
                                corresponding sequence in the batch.

        Returns:
            mx.array: The input tensor with rotary embeddings applied.
        """
        original_shape = x.shape
        x_ndim = x.ndim

        # Reshape x to a canonical (Batch, NumHeads, SeqLen, HeadDims)
        if x_ndim == 2:  # (L, D)
            # This case is tricky with batched offsets.
            # Assuming if L,D and batched offsets, it implies B=1 for x, but offsets define B positions
            # This utility is primarily for already batched x (B, H, L, D) or (B, L, D)
            if offsets.shape[0] != 1:
                # Or, we could try to repeat x for each offset if that's the desired semantic
                raise ValueError("Batched offsets with unbatched input x (L,D) is ambiguous."
                                 "Ensure x has a batch dimension if offsets are batched.")
            x_reshaped = x[None, None, :, :] # Treat as (1, 1, L, D)
            batch_size_x = 1
        elif x_ndim == 3:  # (B, L, D)
            x_reshaped = x[:, None, :, :]
            batch_size_x = x.shape[0]
        elif x_ndim == 4:  # (B, H, L, D)
            x_reshaped = x
            batch_size_x = x.shape[0]
        else:
            raise ValueError(f"Input x must have 2, 3, or 4 dimensions, got {x_ndim}")

        if batch_size_x != offsets.shape[0] and batch_size_x !=1 :
             raise ValueError(
                f"Batch size of x ({batch_size_x}) must match batch size of offsets ({offsets.shape[0]}) or x batch_size must be 1."
            )


        batch_size, num_heads, seq_len, head_dims_x = x_reshaped.shape
        
        if head_dims_x < self.dims:
            raise ValueError(f"Input 'x' head dimension ({head_dims_x}) is less than RoPE 'dims' ({self.dims}). RoPE will only be applied to the first {self.dims} dimensions.")


        # Only apply RoPE to the first self.dims
        x_to_rotate = x_reshaped[..., :self.dims]
        x_remaining = x_reshaped[..., self.dims:]


        # `positions` will be (B, L)
        # `mx.arange` creates (L,), `offsets[:, None]` creates (B,1)
        # Broadcasting `offsets[:, None] + mx.arange(...)` results in (B,L)
        positions = offsets[:, None] + mx.arange(0, seq_len, dtype=mx.float32)

        # Reshape positions for broadcasting with frequencies: (B, 1, L, 1)
        positions_reshaped = positions.reshape(batch_size, 1, seq_len, 1)

        # `computed_frequencies` should be shaped like (1, Dims//2) or (Dims//2,)
        # Reshape for broadcasting: (1, 1, 1, Dims//2)
        if computed_frequencies.ndim == 1: # (Dims//2,)
            freqs_reshaped = computed_frequencies.reshape(1, 1, 1, -1)
        elif computed_frequencies.ndim == 2 and computed_frequencies.shape[0] == 1: # (1, Dims//2)
            freqs_reshaped = computed_frequencies.reshape(1, 1, 1, -1)
        else: # Assume it's already correctly shaped or error
            if computed_frequencies.shape != (1,1,1, self.dims // 2) and computed_frequencies.shape != (self.dims //2):
                 raise ValueError(f"computed_frequencies has unexpected shape {computed_frequencies.shape}")
            freqs_reshaped = computed_frequencies


        # `angles` will be (B, 1, L, Dims//2)
        angles = positions_reshaped * freqs_reshaped

        cos_pos = mx.cos(angles)  # (B, 1, L, Dims//2)
        sin_pos = mx.sin(angles)  # (B, 1, L, Dims//2)

        # These will broadcast over the NumHeads dimension of x_to_rotate
        
        x_even = x_to_rotate[..., ::2]   # (B, H, L, Dims//2)
        x_odd = x_to_rotate[..., 1::2]  # (B, H, L, Dims//2)

        output_even = x_even * cos_pos - x_odd * sin_pos
        output_odd = x_odd * cos_pos + x_even * sin_pos
        
        x_rotated = mx.concatenate((output_even, output_odd), axis=-1) # (B, H, L, Dims)

        # If original head_dims_x was > self.dims, concatenate the non-rotated part
        if self.dims < head_dims_x:
            final_x = mx.concatenate((x_rotated, x_remaining), axis=-1)
        else:
            final_x = x_rotated
            
        return final_x.reshape(original_shape)


# mlx_lm/models/rope_utils.py

# ... other imports ...

class Llama3RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False, # This affects how _freqs is computed
        base: float = 10000,
        scaling_config: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional # Store it if needed for freq computation logic

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )

        # This is the core Llama3 frequency calculation
        # The result `inv_freqs_for_fast_rope` is what mx.fast.rope uses internally when `base` is given.
        # Or, this class computes the final `theta_i` style frequencies.
        # Let's assume self._computed_final_freqs are the `theta_i` values.
        
        # Original Llama3RoPE code from file:
        base_freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims) # These are 1/lambda_i, not theta_i
        wavelens = 2 * mx.pi / base_freqs # Corrected: freqs in hertz if base_freqs are angular speed. Or, 2*pi * base_freqs if base_freqs are freqs.
                                        # The RoPE paper uses `theta_i = 10000^(-2i/d)`. So `base_freqs` here are `theta_i`.

        # The `_freqs` computed here are THE `theta_i`s, ready for `pos * theta_i`.
        # This is what `mx.fast.rope` expects when `freqs` argument is provided.
        # So, `self._final_theta_freqs` will be these.
        
        temp_freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims) # These are the theta_i
        
        scaled_temp_freqs = temp_freqs.copy() # Start with base thetas

        all_wavelens = 2 * mx.pi / temp_freqs # wavelen = 2pi / theta

        # Apply Llama3 scaling logic
        scaled_temp_freqs = mx.where(all_wavelens > (old_context_len / low_freq_factor), scaled_temp_freqs * factor, scaled_temp_freqs)
        
        is_medium_freq = (all_wavelens > (old_context_len / high_freq_factor)) & \
                         (all_wavelens < (old_context_len / low_freq_factor))
        
        smooth_factors = (old_context_len / all_wavelens - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_freqs = temp_freqs / ((1 - smooth_factors) / factor + smooth_factors) # Original freqs scaled

        self._final_theta_freqs = mx.where(is_medium_freq, smooth_freqs, scaled_temp_freqs)
        # Reshape for BatchedRoPEUtility: (1, Dims//2) or (1,1,1,Dims//2)
        self._final_theta_freqs = self._final_theta_freqs.reshape(1, -1)


    def get_computed_frequencies(self) -> mx.array:
        """Returns the final theta_i frequencies for RoPE computation."""
        return self._final_theta_freqs

    def get_input_scale(self) -> float:
        """Llama3RoPE doesn't explicitly scale input x before rotation in its formula."""
        return 1.0

    def extra_repr(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )

    def __call__(self, x, offset: int = 0):
        # This is the original scalar offset path using mx.fast.rope
        # It's efficient for single sequences.
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional, # fast.rope uses traditional to compute freqs from base if self._final_theta_freqs is None
            base=None, # Important: We provide freqs directly
            scale=1.0, # mx.fast.rope's scale applies to positions, not x.
                       # Our self._final_theta_freqs already incorporate scaling factors.
            offset=offset,
            freqs=self._final_theta_freqs, # Pass the pre-computed, scaled thetas
        )

# Apply similar modifications to YarnRoPE and SuScaledRotaryEmbedding:
# - `get_computed_frequencies()`: Should return their `self._freqs` (which are theta_i).
# - `get_input_scale()`:
#   - For YarnRoPE: return `self.mscale`
#   - For SuScaledRotaryEmbedding: return `self.scale`
#
# For the base nn.RoPE:
# nn.RoPE(dims, traditional, base, scale)
# - `get_computed_frequencies()`:
#   if traditional: freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
#   else: freq = mx.exp(mx.arange(0, dims, 2, dtype=mx.float32) * (-mx.log(base) / dims))
#   return freq.reshape(1,-1) # This is theta_i
# - `get_input_scale()`: return 1.0
# - Note: `nn.RoPE`'s `scale` parameter applies to positions. `mx.fast.rope`'s `scale` also applies to positions.
#   If `nn.RoPE.scale` is not 1.0, `BatchedRoPEUtility` would need to incorporate this:
#   `positions = (offsets[:, None] + mx.arange(0, seq_len, dtype=mx.float32)) * nn_rope_scale_factor`


# mlx_lm/models/llama.py (or equivalent model file)

# ... other imports ...
# from .rope_utils import initialize_rope # Assumes this is updated or RoPE classes are.
from .batched_rope import BatchedRoPEUtility # Import the new utility

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args # Store args
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads
        self.scale_sdpa = head_dim**-0.5 # Renamed to avoid conflict with RoPE scale

        # Projections
        attention_bias = getattr(args, "attention_bias", False)
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias)

        # RoPE initialization using existing utility
        self.rope_instance = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling, # This is the dict like {"type": "llama3", "factor": ...}
            args.max_position_embeddings,
        )

        # Instantiate our batched utility
        self.batched_rope_utility = BatchedRoPEUtility(dims=self.head_dim)


    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        per_sequence_offsets: Optional[mx.array] = None, # mx.array for batched
    ) -> mx.array:
        B, L, D = x.shape
        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Determine if we are in batched mode for RoPE based on per_sequence_offsets
        is_batched_rope_mode = isinstance(per_sequence_offsets, mx.array)

        if is_batched_rope_mode:
            # Fetch RoPE parameters from the instantiated RoPE variant
            # These methods need to be added to each RoPE class in rope_utils.py etc.
            if not hasattr(self.rope_instance, "get_computed_frequencies") or \
               not hasattr(self.rope_instance, "get_input_scale"):
                raise AttributeError(
                    f"RoPE instance of type {type(self.rope_instance)} "
                    "does not have get_computed_frequencies or get_input_scale method."
                )
            
            rope_theta_freqs = self.rope_instance.get_computed_frequencies()
            rope_input_x_scale = self.rope_instance.get_input_scale()
            
            # Apply input scaling if necessary
            queries_scaled = queries * rope_input_x_scale if rope_input_x_scale != 1.0 else queries
            keys_scaled = keys * rope_input_x_scale if rope_input_x_scale != 1.0 else keys
            
            # Apply RoPE using the batched utility
            queries_rotated = self.batched_rope_utility(queries_scaled, rope_theta_freqs, per_sequence_offsets)
            keys_rotated = self.batched_rope_utility(keys_scaled, rope_theta_freqs, per_sequence_offsets)

        else: # Scalar offset or no offset (offset=0 by default in RoPE classes if cache is None)
            scalar_offset = per_sequence_offsets if per_sequence_offsets is not None else (cache.offset if cache else 0)
            if not isinstance(scalar_offset, int):
                raise ValueError(f"Offset must be an int for scalar RoPE path, got {type(scalar_offset)}")

            # Use the RoPE instance's own __call__ for scalar path (uses mx.fast.rope)
            queries_rotated = self.rope_instance(queries, offset=scalar_offset)
            keys_rotated = self.rope_instance(keys, offset=scalar_offset)

        if cache is not None:
            # update_and_fetch must be batch-aware if keys_rotated/values are batched
            keys_rotated, values = cache.update_and_fetch(keys_rotated, values)

        output = scaled_dot_product_attention( # Ensure this function is correctly imported/defined
            queries_rotated, keys_rotated, values, scale=self.scale_sdpa, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)

# Remember to also plumb `per_sequence_offsets` through TransformerBlock, LlamaModel, and Model classes
# as shown in the previous response.