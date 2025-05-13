import os
import sys
# Ensure project root is on sys.path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import mlx.core as mx
from mlx_lm.models.base import create_causal_mask, create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.paged_kvcache import PagedKVCache


class DummyCache:
    """A simple cache-like object with offset and max_size attributes for testing."""
    def __init__(self, offset, max_size):
        self.offset = offset
        self.max_size = max_size


def test_create_causal_mask_basic():
    # No offset, no window_size: simple lower-triangular mask
    mask = create_causal_mask(N=3)
    expected = [
        [True,  False, False],
        [True,  True,  False],
        [True,  True,  True ],
    ]
    assert mask.shape == (3, 3)
    assert mask.tolist() == expected


def test_create_causal_mask_with_offset():
    # Offset shifts the window; shape expands to (N, offset+N)
    mask = create_causal_mask(N=2, offset=3)
    # Rows correspond to linds = [3, 4] and rinds = [0,1,2,3,4]
    expected = [
        [True, True, True, True,  False],
        [True, True, True, True,  True ],
    ]
    assert mask.shape == (2, 5)
    assert mask.tolist() == expected


def test_create_causal_mask_with_window_size():
    # Sliding window: limit attention to window_size tokens behind
    mask = create_causal_mask(N=3, offset=2, window_size=2)
    expected = [
        [True,  True,  True,  False, False],
        [False, True,  True,  True,  False],
        [False, False, True,  True,  True ],
    ]
    assert mask.shape == (3, 5)
    assert mask.tolist() == expected


def test_create_attention_mask_no_cache_returns_causal():
    # By default, without return_array or cache, should return the string 'causal'
    h = mx.zeros((2, 4, 8), dtype=mx.float32)
    result = create_attention_mask(h, cache=None)
    assert result == "causal"


def test_create_attention_mask_no_cache_return_array():
    # Request explicit array output without a cache
    h = mx.zeros((1, 3, 8), dtype=mx.float32)
    mask = create_attention_mask(h, cache=None, return_array=True)
    expected = create_causal_mask(N=3)
    assert mask.tolist() == expected.tolist()


def test_create_attention_mask_single_length():
    # Single-token sequence should get no mask
    h = mx.zeros((1, 1, 8), dtype=mx.float32)
    result = create_attention_mask(h, cache=None)
    assert result is None


def test_create_attention_mask_with_kvcache_offset():
    # Simulate a cache that already has 3 tokens
    keys = mx.random.normal((1, 1, 3, 4), dtype=mx.float32)
    values = mx.random.normal((1, 1, 3, 4), dtype=mx.float32)
    kv = KVCache()
    kv.update_and_fetch(keys, values)
    # New segment of length 2
    h = mx.zeros((1, 2, 8), dtype=mx.float32)
    mask = create_attention_mask(h, cache=[kv], return_array=True)
    expected = create_causal_mask(N=2, offset=3)
    assert mask.tolist() == expected.tolist()


def test_create_attention_mask_with_dummy_cache_window():
    # Sliding-window triggered by max_size on cache
    dummy = DummyCache(offset=2, max_size=3)
    h = mx.zeros((1, 2, 8), dtype=mx.float32)
    # Default return_array=False, but offset+T > max_size triggers array path
    mask = create_attention_mask(h, cache=[dummy])
    expected = create_causal_mask(N=2, offset=2, window_size=3)
    assert mask.tolist() == expected.tolist()


def test_create_attention_mask_with_paged_kvcache_single():
    # Single sequence: offset should come from PagedKVCache
    B, H, D = 1, 1, 2
    block_size, num_blocks = 4, 10
    paged = PagedKVCache(num_layers=1, num_heads=H, head_dim=D, num_blocks=num_blocks, block_size=block_size, dtype=mx.float32)
    # register sequence 0
    paged.add_sequence(seq_id=0, max_len=0)
    # append 3 tokens
    keys = mx.arange(H * 3 * D, dtype=mx.float32).reshape((H, 3, D))
    paged.append(0, layer=0, keys=keys, values=keys)
    # now offset == 3
    # generate mask for T=2 new tokens
    h = mx.zeros((B, 2, 4), dtype=mx.float32)
    mask = create_attention_mask(h, cache=[paged])
    expected = create_causal_mask(N=2, offset=3)
    # mask has batch dimension of size 1
    assert mask.shape[0] == 1
    # compare the single sequence's mask
    assert mask[0].tolist() == expected.tolist()


def test_create_attention_mask_with_paged_kvcache_batch():
    # Batch of 2 sequences with different offsets
    B, H, D = 2, 1, 2
    block_size, num_blocks = 5, 10
    paged = PagedKVCache(num_layers=1, num_heads=H, head_dim=D, num_blocks=num_blocks, block_size=block_size, dtype=mx.float32)
    # register two sequences
    paged.add_sequence(seq_id=0, max_len=0)
    paged.add_sequence(seq_id=1, max_len=0)
    # seq 0: append 1 token
    keys0 = mx.arange(H * 1 * D, dtype=mx.float32).reshape((H, 1, D))
    paged.append(0, layer=0, keys=keys0, values=keys0)
    # seq 1: append 2 tokens
    keys1 = (mx.arange(H * 2 * D, dtype=mx.float32) + 10).reshape((H, 2, D))
    paged.append(1, layer=0, keys=keys1, values=keys1)
    # now offsets == [1,2]
    # generate mask for T=3 new tokens
    T = 3
    h = mx.zeros((B, T, 4), dtype=mx.float32)
    mask = create_attention_mask(h, cache=[paged], return_array=True)
    # compute expected per-sequence masks
    offs = paged.offsets
    max_offset = max(offs)
    total_K = max_offset + T
    expected_masks = []
    for off in offs:
        m = create_causal_mask(N=T, offset=off)
        # pad each mask to the unified width
        pad_width = total_K - m.shape[1]
        if pad_width > 0:
            # pad along the key dimension (columns)
            m = mx.pad(m, ((0, 0), (0, pad_width)))
        expected_masks.append(m)
    expected = mx.stack(expected_masks, axis=0)
    assert mask.tolist() == expected.tolist() 