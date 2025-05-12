# test_batcher_phase1.py
import time
import mlx.core as mx
from mlx_lm import load # Assuming mlx_lm is installed or in PYTHONPATH
from mlx_lm.continuous_batcher import ContinuousBatcher # Your new file

def main():
    # model_name = "mlx-community/Llama-3-8B-Instruct-4bit" 
    model_name = "mlx-community/Qwen3-0.6B-4bit-DWQ-05092025" # Smaller model for faster testing
    print(f"Loading model: {model_name}")
    model, tokenizer = load(model_name)
    print("Model loaded.")

    batcher = ContinuousBatcher(model, tokenizer)

    # Submit multiple test requests
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "Once upon a time, in a land far away,",
        "In a galaxy far, far away, there was",
        "To be or not to be, that is the question:",
        "What is the meaning of life?",
        "Data science is",
        "The quick brown fox jumps over the lazy dog",
        "Explain quantum mechanics in simple terms.",
        "List the first ten prime numbers:",
    ]
    req_ids = []
    for p in prompts:
        # Use moderate temperature for variety
        req_ids.append(batcher.submit(p, max_tokens=10, temperature=0.7))

    print(f"Submitted {len(req_ids)} requests: {req_ids}")
    active_requests = set(req_ids)
    
    # Start benchmark timer (exclude model loading/prefill)
    total_tokens = 0
    decode_start = time.perf_counter()

    # Generation loop
    for _iteration in range(20): # Max iterations to prevent infinite loop
        if not batcher.has_active_requests():
            print("All requests finished.")
            break

        print(f"\n--- Iteration {_iteration + 1} ---")
        responses = batcher.step()
        
        for resp in responses:
            print(f"  Request {resp.request_id}: Token ID {resp.token_id}, Segment '{resp.text_segment}', Finished: {resp.finished} ({resp.finish_reason})")
            if resp.finished and resp.request_id in active_requests:
                active_requests.remove(resp.request_id)
        
        # Count the tokens generated this step
        total_tokens += len(responses)
        
        # time.sleep(0.1) # Small delay (disabled for benchmarking)
    else:
        print("Reached max iterations.")

    # End benchmark timer
    decode_end = time.perf_counter()
    elapsed = decode_end - decode_start
    print(f"\nDecoded {total_tokens} tokens in {elapsed:.2f}s => {total_tokens/elapsed:.2f} tokens/sec")
    print("\n--- Final Results ---")
    for rid in req_ids:
        print(f"Request {rid} Full Text: {batcher.get_full_text(rid)}")

if __name__ == "__main__":
    main()