import mlx.core as mx
from typing import Dict, List

class PagedKVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, num_blocks: int, block_size: int, dtype: mx.Dtype):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size # number of tokens per block/pages
        self.dtype = dtype
        self.num_blocks = num_blocks

        # key 
        self.key_cache = mx.zeros((num_layers, num_blocks, num_heads, block_size, head_dim), dtype=dtype)
        # value
        self.value_cache = mx.zeros_like(self.key_cache)
        
        # map sequence id to block ids for each layer
        self.block_tables: Dict[int, List[List[int]]] = {} # seq_id -> [[block_idx layer 1], [block_idx layer 2], ...]

        # for each sequence, the current logical length (offset for RoPE)
        self.seq_lengths: Dict[int, int] = {} # seq_id -> seq_len
        

        # free list of block 
        self.free_blocks: List[int] = list(range(self.num_blocks))

        self.block_ref_counts: List[int] = [0] * self.num_blocks

    def add_sequence(self, seq_id: int, max_len: int):
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already exists")

        # map sequence id to block ids
        self.block_tables[seq_id] = [[] for _ in range(self.num_layers)]

        self.seq_lengths[seq_id] = 0

        # preallocate blocks for this sequence up to max_len
        # ceil of max_len / block_size
        num_blocks = (max_len + self.block_size - 1) // self.block_size

        for layer in range(self.num_layers):
            table = self.block_tables[seq_id][layer]
            for _ in range(num_blocks):
                block_idx = self._alloc_block()
                table.append(block_idx)
                self.block_ref_counts[block_idx] += 1

        return table

    def remove_sequence(self, seq_id: int):
        # return its block to the free list of blocks
        for layer_blocks in self.block_tables[seq_id]:
            for block_idx in layer_blocks:
                self.block_ref_counts[block_idx] -= 1
                if self.block_ref_counts[block_idx] == 0:
                    self._free_block(block_idx)

            

        del self.block_tables[seq_id]
        del self.seq_lengths[seq_id]


    def _alloc_block(self) -> int:
        """Allocate a block from the free list"""
        # first pop from the free list 
        if not self.free_blocks:
            raise MemoryError("PagedKVCache: Out of blocks!") # More specific error
        
        
        block_idx = self.free_blocks.pop()
        self.block_ref_counts[block_idx] = 0 # set to zero because no sequence is using it
        return block_idx
    
    def _free_block(self, block_idx: int):
        """Return a block to the free list"""

        self.free_blocks.append(block_idx)
        self.block_ref_counts[block_idx] = 0 # set to zero because no sequence is using it
        
    def append(self, seq_id: int, layer: int, keys: mx.array, values: mx.array):
        """
        Append to cache. keys and values are [num_heads, seq_len, dim_head]

        """

        # current logical length of the sequence
        length = self.seq_lengths[seq_id]
        table: List[int] = self.block_tables[seq_id][layer] # list of block ids for this sequence and layer

        
        # TODO: vectorize this
        for i in range(keys.shape[1]):
            pos = length + i
            block_num = pos // self.block_size # which block to put the token in
            offset = pos % self.block_size # where in the block to put the token

            # if this sequence hadn't used this block before, allocate it
            if block_num >= len(table):
                block_idx = self._alloc_block()
                table.append(block_idx)
                self.block_ref_counts[block_idx] += 1 # this sequence is now using this block
                

            # get the block id
            block_idx = table[block_num]

            # write into global cache

            self.key_cache[layer, block_idx, :, offset, :] = keys[:, i, :] # LHS: [num_heads, dim_head], RHS: [num_heads, dim_head]
            self.value_cache[layer, block_idx, :, offset, :] = values[:, i, :]
            
        # update logical length
        self.seq_lengths[seq_id] += keys.shape[1]

    def get_kv_attention(self, seq_ids: List[int], layer: int):
        """
        Get the keys and values for the given sequence ids and layer

        Args:
            seq_ids: list of sequence ids
            layer: layer index

        Returns:
            keys: [num_heads, seq_len, dim_head]
            values: [num_heads, seq_len, dim_head]
        """
        # we need to pad the keys and values to the max length
        max_len = max(self.seq_lengths[seq_id] for seq_id in seq_ids)
        
        # list of keys and values for each sequence
        batch_k: List[mx.array] = []
        batch_v: List[mx.array] = []

        for s in seq_ids:
            # get the logical length of the sequence, how many tokens have this sequence produced so far
            L = self.seq_lengths[s]
            
            if L == 0:
                # empty sequence → just zeros of shape [H, max_len, D]
                batch_k.append(mx.zeros((self.num_heads, 0, self.head_dim), dtype=self.dtype))
                batch_v.append(mx.zeros((self.num_heads, 0, self.head_dim), dtype=self.dtype))
                continue
            
            idxs: mx.array = mx.array(self.block_tables[s][layer], dtype=mx.int32)  # [B]

            # 3) Gather those blocks in one shot:
             #    key_cache[layer] has shape [num_blocks, H, block_size, D]
             #    after take -> shape [B, H, block_size, D]
             #    where B is the number of blocks in the sequence

            k_blocks: List[mx.array] = mx.take(self.key_cache[layer], idxs, axis=0)
            v_blocks: List[mx.array] = mx.take(self.value_cache[layer], idxs, axis=0)

            # transpose
            # [B, H, block_size, D] -> [H, B, block_size, D]
            k_blocks = mx.transpose(k_blocks, (1, 0, 2, 3))
            v_blocks = mx.transpose(v_blocks, (1, 0, 2, 3))

            # collapse the pages dimension B and the token-per-page dimension block_size
            k_blocks = mx.reshape(k_blocks, (self.num_heads, -1, self.head_dim))
            v_blocks = mx.reshape(v_blocks, (self.num_heads, -1, self.head_dim))

            # slice of the actual L tokens
            k_blocks = k_blocks[:, :L, :]
            v_blocks = v_blocks[:, :L, :]

            # pad the rest of the tokens with zeros
            pad_len = max_len - L
            k_blocks = mx.pad(k_blocks, ((0, 0), (0, pad_len), (0, 0)))
            v_blocks = mx.pad(v_blocks, ((0, 0), (0, pad_len), (0, 0)))

            batch_k.append(k_blocks)
            batch_v.append(v_blocks)

     

        # stack along axis 0 to get a new dimension
        batch_k = mx.stack(batch_k, axis=0)
        batch_v = mx.stack(batch_v, axis=0)

        return batch_k, batch_v

        
    

    def trim(self, seq_id: int, max_len: int):
        """
        Trim the cache and preserve only the last max_len tokens
        """
        L = self.seq_lengths[seq_id]
        if L <= max_len:
            return
        
        # number of tokens to drop
        drop = L - max_len
        # we need to know the number of blocks to drop and which blocks to drop
        full_pages = drop // self.block_size # this is the number of full pages to drop

        # drop the full pages
        for layer in range(self.num_layers):
            # shape -> [block_idx]
            table = self.block_tables[seq_id][layer]

            free_table = table[:full_pages]
            self.block_tables[seq_id][layer] = table[full_pages:]

            for block_idx in free_table:
                self.block_ref_counts[block_idx] -= 1
                if self.block_ref_counts[block_idx] == 0:
                    self._free_block(block_idx)

        
    
        # Update the sequence length to max len 
        self.seq_lengths[seq_id] = max_len
            



    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new keys/values and return the stacked cache for single-layer usage."""
        # keys: [B, num_heads, T, D], values: [B, num_heads, T, D]
        B = keys.shape[0]
        if self.num_layers != 1:
            raise ValueError(f"PagedKVCache.update_and_fetch only supports num_layers=1, got {self.num_layers}")
        layer_idx = 0
        
        seq_ids = list(self.seq_lengths.keys())
        for i, s in enumerate(seq_ids):
            if s not in self.block_tables:
                # lazily register sequence with no preallocation
                self.add_sequence(s, max_len=0)
            self.append(s, layer_idx, keys[i], values[i])
        k, v = self.get_kv_attention(seq_ids, layer_idx)
        return k, v

    @property
    def offsets(self) -> list:
        return list(self.seq_lengths.values())