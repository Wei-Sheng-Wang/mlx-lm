import os
import sys
# Ensure project root is on sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import mlx.core as mx
from mlx_lm.models.paged_kvcache import PagedKVCache
from mlx_lm.models.cache import KVCache


def test_update_and_fetch_basic_and_incremental():
    # Basic single-sequence append and fetch
    B, H, T, D = 1, 1, 3, 2
    # initialize cache with one layer, head_dim=D, block_size=2, ample num_blocks
    cache = PagedKVCache(num_layers=1, num_heads=H, head_dim=D, num_blocks=10, block_size=2, dtype=mx.float32)
    # Create sequential keys and values so we can verify ordering
    flat = mx.arange(H * T * D, dtype=mx.float32)
    keys0 = flat.reshape((H, T, D))[None, ...]  # shape [1, H, T, D]
    values0 = keys0
    out1, _ = cache.update_and_fetch(keys0, values0)
    # After first fetch, should equal the original keys0
    assert out1.shape == (B, H, T, D)
    assert mx.allclose(out1, keys0)
    # Append another batch of tokens with distinct values
    T2 = 2
    keys1 = mx.full((H, T2, D), 100, dtype=mx.float32)[None, ...]
    values1 = keys1
    out2, _ = cache.update_and_fetch(keys1, values1)
    # Now total length is T + T2 == 5
    assert out2.shape == (B, H, T + T2, D)
    # First T tokens should match keys0, next T2 tokens should match keys1
    assert mx.allclose(out2[:, :, :T, :], keys0)
    assert mx.allclose(out2[:, :, T:, :], keys1)


def test_multi_sequence_and_offsets_and_remove():
    # Test multiple sequences in a single-layer cache
    B, H, T, D = 2, 1, 2, 2
    cache = PagedKVCache(num_layers=1, num_heads=H, head_dim=D, num_blocks=10, block_size=4, dtype=mx.float32)
    # Sequence 0 values [0,1,2,3] and sequence 1 values [10,11,12,13]
    base0 = mx.arange(H * T * D, dtype=mx.float32).reshape((H, T, D))
    base1 = base0 + 10
    keys = mx.stack([base0, base1], axis=0)  # shape [2, H, T, D]
    values = keys
    out, _ = cache.update_and_fetch(keys, values)
    assert out.shape == (B, H, T, D)
    # Check that each sequence returns its own data
    assert mx.allclose(out[0], keys[0])
    assert mx.allclose(out[1], keys[1])
    # Offsets should record T tokens each
    offs = cache.offsets
    assert offs[0] == T
    assert offs[1] == T
    arr = cache.get_offsets_for_batch([0, 1])
    assert arr.tolist() == [T, T]
    # The single-sequence offset property returns seq 0 length
    assert cache.offset == T
    # Test remove_sequence
    cache.add_sequence(seq_id=5, max_len=1)
    assert 5 in cache.block_tables
    cache.remove_sequence(5)
    assert 5 not in cache.block_tables
    assert 5 not in cache.seq_lengths


def test_trim_preserves_last_tokens():
    # Test trimming drops full blocks and preserves last max_len tokens
    B, H, T, D = 1, 1, 3, 2
    cache = PagedKVCache(num_layers=1, num_heads=H, head_dim=D, num_blocks=10, block_size=2, dtype=mx.float32)
    # keys seq: values [[0,1],[2,3],[4,5]] for tokens 0,1,2
    flat = mx.arange(H * T * D, dtype=mx.float32)
    keys = flat.reshape((H, T, D))[None, ...]
    values = keys
    out_full, _ = cache.update_and_fetch(keys, values)
    assert out_full.shape == (B, H, T, D)
    # Trim to only keep last 1 token (token index 2)
    cache.trim(seq_id=0, max_len=1)
    # After trimming, only one token remains
    arr = cache.get_offsets_for_batch([0])
    assert arr.tolist() == [1]
    trimmed_k, trimmed_v = cache.get_kv_attention([0], layer=0)
    # Should match out_full for token index 2 only
    expected = out_full[:, :, 2:3, :]
    assert mx.allclose(trimmed_k, expected)
    assert mx.allclose(trimmed_v, expected)


def test_equivalence_to_kvcache_random_chunks():
    # Ensure PagedKVCache matches the baseline KVCache for chunked updates
    B, H, D = 2, 2, 4
    block_size = 3
    num_blocks = 10
    paged = PagedKVCache(
        num_layers=1,
        num_heads=H,
        head_dim=D,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=mx.float32,
    )
    kv = KVCache()

    # Simulate feeding in several chunks of varying lengths
    chunks = [2, 3, 1]
    total_length = 0
    for chunk in chunks:
        # Random keys and values for this chunk
        keys = mx.random.normal((B, H, chunk, D), dtype=mx.float32)
        values = mx.random.normal((B, H, chunk, D), dtype=mx.float32)
        # Update both caches
        k_baseline, v_baseline = kv.update_and_fetch(keys, values)
        k_paged, v_paged = paged.update_and_fetch(keys, values)

        # They should have the same shape and contents
        assert k_paged.shape == k_baseline.shape
        assert v_paged.shape == v_baseline.shape
        assert mx.allclose(k_paged, k_baseline)
        assert mx.allclose(v_paged, v_baseline)

        total_length += chunk

    # Offsets/tracking length should agree
    assert kv.offset == total_length
    assert paged.offset == total_length


@pytest.mark.parametrize("B,H,D,block_size,num_blocks,chunks", [
    (1, 1, 3, 2, 5, [1, 2, 3]),
    (2, 2, 4, 3, 10, [2, 3, 1]),
    (3, 1, 5, 4, 8, [1, 4, 2]),
])
def test_equivalence_various_params(B, H, D, block_size, num_blocks, chunks):
    """
    Test PagedKVCache against KVCache under various tensor shapes and block configurations.
    """
    paged = PagedKVCache(
        num_layers=1,
        num_heads=H,
        head_dim=D,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=mx.float32,
    )
    kv = KVCache()

    total_length = 0
    for chunk in chunks:
        # Generate synthetic keys and values for this chunk
        keys = mx.random.normal((B, H, chunk, D), dtype=mx.float32)
        values = mx.random.normal((B, H, chunk, D), dtype=mx.float32)
        # Update both caches
        k_base, v_base = kv.update_and_fetch(keys, values)
        k_page, v_page = paged.update_and_fetch(keys, values)

        # Assert shapes and content match
        assert k_page.shape == k_base.shape
        assert v_page.shape == v_base.shape
        assert mx.allclose(k_page, k_base)
        assert mx.allclose(v_page, v_base)

        total_length += chunk

    # Finally, offsets should be tracked equally
    assert kv.offset == total_length
    assert paged.offset == total_length 