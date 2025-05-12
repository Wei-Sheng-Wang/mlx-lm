from typing import Optional

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
        if dims % 2 != 0:
            raise ValueError(f"RoPE dims must be even, got {dims}")
        self.dims = dims # The number of dimensions to rotate (typically head_dim)

    def __call__(
        self,
        x: mx.array,
        frequencies: mx.array,
        offsets: mx.array,
        position_scale_factor: float = 1.0, # New: To handle nn.RoPE's scale
    ) -> mx.array:
        """
        Apply RoPE with batched offsets.

        Args:
            x (mx.array): Input tensor. Expected shape (B, NumHeads, SeqLen, HeadDims).
                          Can also handle (B, SeqLen, HeadDims) or (SeqLen, HeadDims)
                          which will be reshaped internally.
            frequencies (mx.array): The pre-computed inverse frequencies (theta_i).
                                           Expected shape like (1, Dims//2) or broadcastable
                                           to (1, 1, 1, Dims//2) for `angles` calculation.
            offsets (mx.array): A 1D array of per-sequence offsets. Shape (B,).
                                Each offset is the starting position for the
                                corresponding sequence in the batch.
            position_scale_factor (float): Scale factor for positions, 
                                           compatible with nn.RoPE's `scale` param.

        Returns:
            mx.array: The input tensor with rotary embeddings applied.
        """
        original_shape = x.shape
        x_ndim = x.ndim

        if x_ndim == 2:
            if offsets.shape[0] != 1:
                raise ValueError(
                    "Batched offsets with unbatched input x (L,D) is ambiguous. "
                    "Ensure x has a batch dimension if offsets are batched or B=1 for offsets."
                )
            x_reshaped = x[None, None, :, :] # reshape to (1,1,L,D)
            batch_size_x = 1 # 1 batch item
        elif x_ndim == 3:  
            x_reshaped = x[:, None, :, :] # reshape to (B,1,L,D)
            batch_size_x = x.shape[0] # B batch items
        elif x_ndim == 4:  
            x_reshaped = x
            batch_size_x = x.shape[0] 
        else:
            raise ValueError(f"Input x must have 2, 3, or 4 dimensions, got {x_ndim}")

        # Validate batch compatibility
        if batch_size_x != 1 and offsets.shape[0] != 1 and batch_size_x != offsets.shape[0]:
             raise ValueError(
                f"Batch size of x ({batch_size_x}) must match batch size of offsets ({offsets.shape[0]}) "
                "or one of them must be 1."
            )
        
        # Determine effective batch size for broadcasting positions and angles
        # This batch_size will be used for operations involving offsets and frequencies
        effective_batch_size = max(batch_size_x, offsets.shape[0])

        num_heads, seq_len, head_dims_x = x_reshaped.shape[1], x_reshaped.shape[2], x_reshaped.shape[-1]
        
        if self.dims > head_dims_x:
            raise ValueError(
                f"RoPE dims ({self.dims}) cannot be greater than "
                f"the input's last dimension ({head_dims_x})."
            )
        
        # Prepare offsets: (effective_batch_size, 1)
        # offsets is initially (B_offset,)
        if offsets.ndim == 1:
            if offsets.shape[0] == 1 and effective_batch_size > 1:
                offsets = mx.repeat(offsets, effective_batch_size, axis=0)
            # After potential repeat, offsets.shape[0] should align with effective_batch_size
            # if it was the determining factor or was repeated.
            offsets = offsets.reshape(effective_batch_size, 1)
        elif offsets.shape[0] != effective_batch_size or offsets.ndim != 2 or offsets.shape[1] != 1:
            # If offsets is not 1D, it must already be (effective_batch_size, 1)
            raise ValueError(
                f"Offsets shape {offsets.shape} is not compatible with effective batch size {effective_batch_size}."
            )

        raw_positions = mx.arange(0, seq_len, dtype=mx.float32) # (L,)
        
        positions = (offsets + raw_positions) # (EffectiveBatch, SeqLen)
        if position_scale_factor != 1.0:
            positions *= position_scale_factor

        # Reshape positions for broadcasting with frequencies: (EffectiveBatch, 1, SeqLen, 1)
        positions_reshaped = positions.reshape(effective_batch_size, 1, seq_len, 1)
        
        # frequencies: (1, Dims//2) or (1,1,1, Dims//2)
        # Ensure frequencies is (1,1,1, self.dims//2)
        if frequencies.ndim == 1 and frequencies.shape[0] == self.dims // 2 or \
            frequencies.ndim == 2 and frequencies.shape == (1, self.dims // 2):
            frequencies = frequencies.reshape(1, 1, 1, -1)
        elif frequencies.shape != (1, 1, 1, self.dims // 2):
            raise ValueError(
                f"frequencies has unexpected shape {frequencies.shape}. "
                f"Expected (self.dims//2,), (1, self.dims//2), or (1,1,1, self.dims//2)."
            )

        angles = positions_reshaped * frequencies # (EffectiveBatch, 1, SeqLen, Dims//2)
        
        cos_pos = mx.cos(angles) # Shape: (EffectiveBatch, 1, SeqLen, Dims//2)
        sin_pos = mx.sin(angles) # Shape: (EffectiveBatch, 1, SeqLen, Dims//2)

        # x_reshaped is (BatchX, NumHeads, SeqLen, HeadDimsX)
        # RoPE is applied to the first self.dims of HeadDimsX
        x_to_rotate = x_reshaped[..., :self.dims]     # (BatchX, NumHeads, SeqLen, self.dims)
        x_pass_through = x_reshaped[..., self.dims:] # (BatchX, NumHeads, SeqLen, HeadDimsX - self.dims)

        x_rot_even = x_to_rotate[..., ::2]    # (BatchX, NumHeads, SeqLen, self.dims//2)
        x_rot_odd = x_to_rotate[..., 1::2]   # (BatchX, NumHeads, SeqLen, self.dims//2)

        # Perform rotation. Broadcasting occurs here:
        # cos_pos/sin_pos (EffectiveBatch, 1, ...) with x_rot_even/odd (BatchX, NumHeads, ...)
        # Resulting shape e.g. (EffectiveBatch, NumHeads, SeqLen, self.dims//2)
        rotated_x_even = x_rot_even * cos_pos - x_rot_odd * sin_pos
        rotated_x_odd = x_rot_odd * cos_pos + x_rot_even * sin_pos # Typo fixed: x_rot_even_sin_pos + x_rot_odd_cos_pos
        
        # Interleave the rotated parts correctly
        # The shape of rotated_x_even/odd already reflects broadcasting from cos_pos/sin_pos
        final_rotated_part_shape_prefix = rotated_x_even.shape[:-1]
        final_rotated_part_dims = self.dims
        
        # Allocate memory for the fully rotated part with the correct (possibly broadcasted) shape
        final_rotated_part = mx.zeros(
            final_rotated_part_shape_prefix + (final_rotated_part_dims,), 
            dtype=x.dtype
        )
        final_rotated_part[..., ::2] = rotated_x_even
        final_rotated_part[..., 1::2] = rotated_x_odd
            
        # Concatenate with the pass-through part, if any
        if self.dims < head_dims_x:
            # x_pass_through might need broadcasting to match final_rotated_part's batch/head dimensions
            # Target shape for broadcasting x_pass_through:
            broadcast_target_shape_pass = final_rotated_part.shape[:-1] + (x_pass_through.shape[-1],)
            x_pass_through_b = mx.broadcast_to(x_pass_through, broadcast_target_shape_pass)
            x_output_reshaped = mx.concatenate((final_rotated_part, x_pass_through_b), axis=-1)
        else: # self.dims == head_dims_x
            x_output_reshaped = final_rotated_part
            
        return x_output_reshaped.reshape(original_shape)

    

# New BatchedRoPEAdapter class
class BatchedRoPEAdapter(nn.Module):
    """
    Adapter class to make BatchedRoPEUtility compatible with the existing
    RoPE module interface expected by Attention layers (i.e., __call__(x, offset=0)).
    
    It pre-calculates/stores frequencies and scaling factors and uses an
    internal BatchedRoPEUtility instance.
    """
    def __init__(
        self,
        dims: int,
        frequencies: mx.array,
        position_scale_factor: float = 1.0,
        yarn_mscale_factor: Optional[float] = None,
        traditional_scaling: bool = False, # For nn.RoPE like scaling of x if mscale is not from YARN
    ):
        super().__init__()
        self.dims = dims
        # Store frequencies with a leading underscore to prevent MLX from treating it as a parameter to be loaded
        self._frequencies = frequencies 
        self.position_scale_factor = position_scale_factor
        self.yarn_mscale_factor = yarn_mscale_factor
        self.traditional_scaling = traditional_scaling

        self.apply_yarn_mscale = (
            self.yarn_mscale_factor is not None and self.yarn_mscale_factor != 1.0
        )
        
        self._batched_rope_utility = BatchedRoPEUtility(dims=dims)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        """
        Args:
            x (mx.array): Input tensor.
            offsets (mx.array or list[int]): Positional offset for the sequence.

        Returns:
            mx.array: The input tensor with rotary embeddings applied.
        """
        # BatchedRoPEUtility expects offsets as a 1D array (batch_size,)
        # For a single sequence (batch_size=1) from typical Attention layer usage, this is (1,)
        offsets = mx.array(offset) if isinstance(offset, list)
        x_to_rope = x
        
        if self.apply_yarn_mscale:
            # YarnRoPE applies mscale internally before rotation.
            # We replicate this by scaling x before passing to BatchedRoPEUtility.
            x_to_rope = x_to_rope[..., :self.dims] * self.yarn_mscale_factor
            if self.dims < x.shape[-1]:
                 x_to_rope = mx.concatenate((x_to_rope, x[..., self.dims:]), axis=-1)
        elif self.traditional_scaling and self.position_scale_factor != 1.0 and not self.apply_yarn_mscale:
            # This handles the case for nn.RoPE where x is scaled by `scale` if `traditional=True`.
            # BatchedRoPEUtility's `position_scale_factor` handles the position scaling, 
            # but nn.RoPE with traditional=True also scales `x` itself.
            # SuScaledRotaryEmbedding also does this.
            # We only apply this if not using YARN mscale, as YARN handles its own scaling.
            x_to_rope = x_to_rope[..., :self.dims] * self.position_scale_factor
            if self.dims < x.shape[-1]:
                 x_to_rope = mx.concatenate((x_to_rope, x[..., self.dims:]), axis=-1)

        return self._batched_rope_utility(
            x_to_rope,
            frequencies=self._frequencies,
            offsets=offsets,
            position_scale_factor=self.position_scale_factor
        )

    
