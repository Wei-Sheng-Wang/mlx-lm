# Copyright © 2023-2024 Apple Inc.

import inspect
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
from mlx.utils import tree_map

from .cache import QuantizedKVCache
from .paged_kvcache import PagedKVCache


@dataclass
class BaseModelArgs:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    lengths: Optional[mx.array] = None,
):
    """
    Create a causal mask for sequence length N with an optional offset and window.
    Supports scalar offset or batched offsets (list or mx.array of shape (B,)).
    """
    # Batched offsets path
    if isinstance(offset, (list, tuple)) or (hasattr(offset, 'ndim') and getattr(offset, 'ndim', 0) > 0):
        offsets = mx.array(offset) if not isinstance(offset, mx.ndarray) else offset
        B = offsets.shape[0]
        base = mx.arange(N)
        # rinds[b,j] = offsets[b] + j
        rinds = offsets[:, None] + base[None, :]
        # linds[b,i] = offsets[b] + i
        linds = offsets[:, None] + base[None, :]
        mask = linds[:, :, None] >= rinds[:, None, :]
        if window_size is not None:
            ws = window_size if isinstance(window_size, (list, tuple, mx.ndarray)) else [window_size] * B
            ws_arr = mx.array(ws) if not isinstance(ws, mx.ndarray) else ws
            mask &= (linds[:, :, None] <= rinds[:, None, :] + ws_arr[:, None, None])
        if lengths is not None:
            mask &= (rinds[:, :, None] < lengths[:, None, None])
        return mask
    # Scalar offset path
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else mx.arange(N)
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask &= (linds <= rinds + window_size)
    if lengths is not None:
        lengths_arr = lengths[:, None, None, None] if isinstance(lengths, mx.ndarray) else lengths
        mask &= (rinds < lengths_arr)
    return mask


def create_attention_mask(
    h: mx.array,  # shape: (B, T, D)
    cache: Optional[Any] = None, 
    return_array: bool = False, 
):
    B, T, _ = h.shape
    if T <= 1:
        return None
    # PagedKVCache batched offsets mask
    if cache is not None and cache[0] is not None and isinstance(cache[0], PagedKVCache):
        paged = cache[0]
        offsets = mx.array(paged.offsets)
        max_offset = int(mx.max(offsets).item())
        total_K = max_offset + T
        # global key positions [0 .. total_K-1]
        rinds = mx.arange(total_K)[None, :]
        # new token positions per sequence
        linds = offsets[:, None] + mx.arange(T)[None, :]
        # broadcast to compute causal mask: (B, T, total_K)
        mask = linds[:, :, None] >= rinds[None, :, :]
        # add head axis so mask is (B,1,T,K) and broadcasts over heads
        mask = mask[:, None, :, :]
        return mask
    # Fallback for scalar-offset caches
    offset = 0
    window_size = None
    if cache is not None and cache[0] is not None:
        c = cache[0]
        offset = c.offset
        if hasattr(c, "max_size"):
            window_size = c.max_size
            offset = min(window_size, offset)
            return_array = return_array or (offset + T > window_size)
    if return_array:
        return create_causal_mask(T, offset, window_size=window_size)
    return "causal"


def quantized_scaled_dot_product_attention(
    queries: mx.array,
    q_keys: tuple[mx.array, mx.array, mx.array],
    q_values: tuple[mx.array, mx.array, mx.array],
    scale: float,
    mask: Optional[mx.array],
    group_size: int = 64,
    bits: int = 8,
) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    if mask is not None:
        if isinstance(mask, str):
            qL, kL = scores.shape[-2:]
            q_indices = mx.arange(kL - qL, kL)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]
        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
        else:
            scores += mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )

    if n_repeats > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))

    return out


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    if isinstance(cache, QuantizedKVCache):
        return quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )



