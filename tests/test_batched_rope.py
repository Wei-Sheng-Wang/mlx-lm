import os
import sys
# Ensure project root is on sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import mlx.core as mx
from mlx_lm.models.batched_rope import BatchedRoPEUtility
from mlx_lm.models.rope_utils import initialize_rope


def test_batched_matches_individual_sequences():
    dims = 6
    B, H, L = 3, 2, 5
    D = dims
    util = BatchedRoPEUtility(D)

    # Create random input of shape (B, H, L, D)
    x = mx.random.normal((B, H, L, D), dtype=mx.float32)
    freqs = mx.arange(D // 2, dtype=mx.float32)
    offsets = mx.array([0, 1, 2], dtype=mx.float32)

    # Batched output
    out_batched = util(x, freqs, offsets)

    # Individual outputs concatenated
    individual_outputs = []
    for i in range(B):
        xi = x[i : i + 1]  # shape (1, H, L, D)
        off_i = mx.array([offsets[i].item()], dtype=mx.float32)
        out_i = util(xi, freqs, off_i)
        individual_outputs.append(out_i)
    out_individual = mx.concatenate(individual_outputs, axis=0)

    # They should be identical
    assert mx.allclose(out_batched, out_individual), (
        "Batched output and individual sequence outputs differ"
    )


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_reshape_equivalence(ndim):
    dims = 8
    L = 7
    D = dims
    util = BatchedRoPEUtility(D)
    freqs = mx.arange(D // 2, dtype=mx.float32)
    offsets = mx.array([5], dtype=mx.float32)

    if ndim == 2:
        # (L, D)
        x2d = mx.random.normal((L, D), dtype=mx.float32)
        out2d = util(x2d, freqs, offsets)
        out4d = util(x2d[None, None, :, :], freqs, offsets).reshape(L, D)
        assert mx.allclose(out2d, out4d)
    elif ndim == 3:
        # (B, L, D)
        B = 1
        x3d = mx.random.normal((B, L, D), dtype=mx.float32)
        out3d = util(x3d, freqs, offsets)
        out4d = util(x3d[:, None, :, :], freqs, offsets).reshape(B, L, D)
        assert mx.allclose(out3d, out4d)
    else:
        # Already 4D (B=1, H=1)
        B, H = 1, 1
        x4d = mx.random.normal((B, H, L, D), dtype=mx.float32)
        out4d = util(x4d, freqs, offsets)
        assert out4d.shape == (B, H, L, D)


def test_freq_shapes_equivalent():
    dims = 10
    util = BatchedRoPEUtility(dims)
    B, H, L = 1, 1, 5
    D = dims
    x = mx.random.normal((B, H, L, D), dtype=mx.float32)

    # Test 1D vs 2D frequency inputs
    freqs_1d = mx.arange(D // 2, dtype=mx.float32)
    freqs_2d = freqs_1d[None, :]
    offsets = mx.array([0], dtype=mx.float32)

    out1 = util(x, freqs_1d, offsets)
    out2 = util(x, freqs_2d, offsets)

    assert mx.allclose(out1, out2), "1D and 2D frequency shapes should produce same output"


def test_batched_matches_default_rope():
    dims = 8
    base = 10000.0
    # Instantiate the default RoPE from rope_utils (nn.RoPE)
    rope = initialize_rope(
        dims=dims,
        base=base,
        traditional=False,
        scaling_config=None,
        max_position_embeddings=None,
    )
    # Extract frequencies if available
    freqs = None
    if hasattr(rope, 'inv_freq'):
        freqs = rope.inv_freq
    elif hasattr(rope, '_freqs'):
        freqs = rope._freqs
    else:
        import pytest; pytest.skip("Default rope does not expose frequencies for comparison")

    util = BatchedRoPEUtility(dims)
    B, H, L = 1, 2, 5
    x = mx.random.normal((B, H, L, dims), dtype=mx.float32)
    offset = 7
    offsets = mx.array([offset], dtype=mx.float32)

    # Compare batched utility to default rope for batch size 1
    out_util = util(x, freqs, offsets)
    out_rope = rope(x, offset=offset)
    assert mx.allclose(out_util, out_rope), "BatchedRoPEUtility differs from default rope implementation" 