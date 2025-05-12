# Copyright © 2023-2024 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

# Import the new adapter
from .batched_rope import BatchedRoPEAdapter


class Llama3RoPE(nn.Module):

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def extra_repr(self):
        return (
            f"{self.dims}, traditional={self.traditional}, "
            f"max_position_embeddings={self.max_position_embeddings}"
        )

    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    def __init__(
        self,
        dims,
        traditional=False,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001  # Prevent singularity

            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (
                max_val - min_val
            )
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * base ** (
            mx.arange(0, dims, 2, dtype=mx.float32) / dims
        )
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x, offset=0):
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims,
    base,
    traditional,
    scaling_config: Optional[dict] = None,
    max_position_embeddings: Optional[int] = None,
):
    if scaling_config is not None:
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
    else:
        rope_type = "default"

    # Temporary RoPE instance for parameter extraction
    temp_rope_instance = None
    yarn_mscale_factor = None
    traditional_scaling_for_adapter = False # Flag for nn.RoPE-like x scaling
    freqs_for_adapter = None
    position_scale_factor_for_adapter = 1.0

    if rope_type in ["default", "linear"]:
        # For mlx.nn.RoPE (default or linear scaling)
        # nn.RoPE calculates inv_freq from `base` and applies `scale` to positions.
        # If traditional=True, it also scales x by `scale`.
        # BatchedRoPEUtility uses `frequencies` (direct freqs) and `position_scale_factor`.
        actual_base = base
        freqs_for_adapter = 1.0 / (actual_base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        
        position_scale_factor_for_adapter = 1.0
        if rope_type == "linear":
            if scaling_config and "factor" in scaling_config:
                # nn.RoPE's `scale` is 1/factor for linear scaling of positions
                position_scale_factor_for_adapter = 1.0 / scaling_config["factor"]
            else:
                raise ValueError("Linear scaling RoPE requires a 'factor' in scaling_config.")
        
        # If traditional is True, nn.RoPE scales x by its `scale` parameter.
        # The adapter needs to know to do this if not YARN mscale.
        if traditional:
            traditional_scaling_for_adapter = True 
            # The `position_scale_factor_for_adapter` is already set correctly for position scaling.
            # The adapter will use this same factor to scale x if traditional_scaling_for_adapter is true.

    elif rope_type == "llama3":
        temp_rope_instance = Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional, # Llama3RoPE doesn't use traditional flag in its math but pass for consistency
            base=base,
            scaling_config=scaling_config,
        )
        freqs_for_adapter = temp_rope_instance._freqs
        # Llama3RoPE uses mx.fast.rope with scale=1.0, positions are scaled via _freqs
        position_scale_factor_for_adapter = 1.0 

    elif rope_type == "yarn":
        scaling_factor = scaling_config["factor"]
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in scaling_config
        }
        temp_rope_instance = YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional, # YarnRoPE doesn't use traditional flag in its math but pass for consistency
            base=base,
            scaling_factor=scaling_factor, # Pass the main scaling_factor here
            **rope_kwargs,
        )
        freqs_for_adapter = temp_rope_instance._freqs
        yarn_mscale_factor = temp_rope_instance.mscale
        # YarnRoPE uses mx.fast.rope with scale=1.0, positions are scaled via _freqs
        # mscale is handled separately for x
        position_scale_factor_for_adapter = 1.0 
    else:
        raise ValueError(f"Unsupported RoPE type {rope_type}")

    return BatchedRoPEAdapter(
        dims=dims,
        frequencies=freqs_for_adapter,
        position_scale_factor=position_scale_factor_for_adapter,
        yarn_mscale_factor=yarn_mscale_factor,
        traditional_scaling=traditional_scaling_for_adapter
    )
