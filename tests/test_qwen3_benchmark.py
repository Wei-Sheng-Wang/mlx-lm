#!/usr/bin/env python3
"""
Benchmark for Qwen3 model: compares batched PagedKVCache inference vs sequential single-prompt inference.
Run with: python tests/test_qwen3_benchmark.py
"""
import os
import sys
import time

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache


def main():
    # Load pretrained model and tokenizer
    model_name = "mlx-community/Qwen3-0.6B-4bit-DWQ-05092025"
    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)
    model.eval()
    # Create a sampler for generation (e.g. nucleus sampling)
    sampler = make_sampler(temp=1.0, top_p=0.9, top_k=0)

    # Prepare text prompts
    prompts = ["how are you?"] * 8
    B = len(prompts)
    # Tokenize prompts (all same length)
    encodings = [tokenizer.encode(p, add_special_tokens=True) for p in prompts]
    T = len(encodings[0])
    inputs = mx.array(encodings, dtype=mx.int32)
    G = 50  # max number of tokens to generate
    print("Prompts:")
    for i, p in enumerate(prompts):
        print(f"  seq {i}: {p}")

    # Benchmark batched inference with PagedKVCache
    use_paged = True
    block_size = 32
    num_blocks = 1000
    cache_batch = make_prompt_cache(
        model,
        max_kv_size=None,
        use_paged_kvcache=use_paged,
        block_size=block_size,
        num_blocks=num_blocks,
    )
    # Register batch sequences so PagedKVCache.offsets is non-empty
    for c in cache_batch:
        for seq_id in range(B):
            c.add_sequence(seq_id, max_len=T)

    print(f"Running batched inference (batch_size={B}, seq_len={T})...")
    start_wall = time.time()
    start_cpu = time.process_time()
    # set model to eval mode, then prefill cache
    _ = model(inputs, cache=cache_batch)
    
    
    wall_batch = time.time() - start_wall
    cpu_batch = time.process_time() - start_cpu
    print(f"Batched: wall_time={wall_batch:.4f}s, cpu_time={cpu_batch:.4f}s")
    # first, report prompt throughput
    total_prompt_tokens = B * T
    prompt_tps = total_prompt_tokens / wall_batch
    print(f"Batched prompt throughput: {prompt_tps:.2f} tokens/sec")
    # now do batched generation
    y = inputs[:, -1:]
    gen_tokens = []
    start_wall_gen = time.time()
    start_cpu_gen = time.process_time()
    for _ in range(G):
        logits = model(y, cache=cache_batch)
        # sample the next token instead of argmax
        next_token = sampler(logits[:, -1, :])
        gen_tokens.append(next_token.tolist())
        y = next_token.reshape(B, 1)
    wall_gen = time.time() - start_wall_gen
    cpu_gen = time.process_time() - start_cpu_gen
    total_gen_tokens = B * G
    gen_tps = total_gen_tokens / wall_gen
    print(f"Batched generation ({G} tokens): wall_time={wall_gen:.4f}s, cpu_time={cpu_gen:.4f}s")
    print(f"Batched generation throughput: {gen_tps:.2f} tokens/sec")
    gen_seqs = [list(seq) for seq in zip(*gen_tokens)]
    print("Batched generated text per sequence:")
    for i, seq_tokens in enumerate(gen_seqs):
        text = tokenizer.decode(seq_tokens, skip_special_tokens=True)
        print(f"  seq {i}: {text}")

    # Benchmark sequential single-item inference
    wall_seq_prompt = 0.0
    cpu_seq_prompt = 0.0
    wall_seq_gen = 0.0
    cpu_seq_gen = 0.0
    print(f"Running sequential inference one-by-one...")
    seq_gen_tokens = []
    for i in range(B):
        # Reuse model; reset cache per sequence
        inp = inputs[i:i+1, :]  # shape (1, T)
        cache_i = make_prompt_cache(
            model,
            max_kv_size=None,
            use_paged_kvcache=use_paged,
            block_size=block_size,
            num_blocks=num_blocks,
        )
        # Register single sequence so PagedKVCache.offsets is non-empty
        for c in cache_i:
            c.add_sequence(seq_id=0, max_len=T)

        # prompt processing
        sw = time.time()
        sc = time.process_time()
        _ = model(inp, cache=cache_i)
        wall_seq_prompt += (time.time() - sw)
        cpu_seq_prompt += (time.process_time() - sc)
        # generation
        y = inp[:, -1:]
        tokens = []
        start_wall_loop = time.time()
        start_cpu_loop = time.process_time()
        for _ in range(G):
            logits = model(y, cache=cache_i)
            # sample the next token
            next_token = sampler(logits[:, -1, :])
            tokens.append(int(next_token.item()))
            y = next_token.reshape(1, 1)
        wall_seq_gen += (time.time() - start_wall_loop)
        cpu_seq_gen += (time.process_time() - start_cpu_loop)
        seq_gen_tokens.append(tokens)
    # report sequential prompt throughput
    total_prompt_tokens = B * T
    prompt_tps_seq = total_prompt_tokens / wall_seq_prompt
    print(f"Sequential prompt total: wall_time={wall_seq_prompt:.4f}s, cpu_time={cpu_seq_prompt:.4f}s")
    print(f"Sequential prompt throughput: {prompt_tps_seq:.2f} tokens/sec")
    # report sequential generation throughput
    total_gen_tokens = B * G
    gen_tps_seq = total_gen_tokens / wall_seq_gen
    print(f"Sequential generation ({G} tokens each): total wall_time={wall_seq_gen:.4f}s, cpu_time={cpu_seq_gen:.4f}s")
    print(f"Sequential generation throughput: {gen_tps_seq:.2f} tokens/sec")
    print("Sequential generated text per sequence:")
    for i, tokens in enumerate(seq_gen_tokens):
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  seq {i}: {text}")


if __name__ == '__main__':
    main() 