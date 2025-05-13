import time
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .models import cache
from .sample_utils import make_sampler
from .tokenizer_utils import TokenizerWrapper
from .models.cache import make_prompt_cache
from .models.qwen3 import create_attention_mask

@dataclass
class AsyncResponse:
    """User-facing response for each generated token"""
    request_id: int # can change to a string if we want to
    token_id: int
    logprobs: mx.array # for later
    text_segment: str
    finished: bool
    finish_reason: Optional[str] = None # length or stop   
    # add timing/stats later if needed

            


@dataclass
class _RequestState:
    """Internal state for managing a single request in the batcher."""
    request_id: int
    prompt_tokens: mx.array
    detokenizer: Any # Instance of tokenizer.detokenizer

   
    max_output_tokens: int 


    # For generation
    next_input_token_id: int # The token to feed into the model next
    kv_cache: Any = None # Will hold the KVCache for this request
    
    finished: bool = False
    finish_reason: Optional[str] = None
    
    # Sampler arguments for this request (can be set via submit)
    temperature: float = 0.0
    top_p: float = 1.0
    # ... other sampler params if needed per request ...

    # Store the full generated text and token_ids for convenience
    full_text: str = ""
    all_generated_token_ids: List[int] = field(default_factory=list)

    tokens_generated: int = 0

    @property
    def logical_length(self) -> int:
        return self.prompt_tokens.size + self.tokens_generated


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
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(tokenizer, TokenizerWrapper):
            tokenizer = TokenizerWrapper(tokenizer)

        self.model = model
        self.tokenizer = tokenizer
        self.requests: Dict[int, _RequestState] = {}
        self._next_request_id_counter = itertools.count()


        self.xtc_probability = xtc_probability
        self.xtc_threshold = xtc_threshold
        self.min_tokens_to_keep = min_tokens_to_keep
        print("ContinuousBatcher initialized (Phase 1: Single request processing).")

    def submit(
        self,
        prompt: Union[str, List[int], mx.array],
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        
    ) -> int:
        """
        Register a new generation job and return its internal request_id. Init
        this will perform a single prefill for this request
        """
        max_kv_size = 1024
        block_size = 16
        num_blocks = 64

        # encode the prompt first
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        elif isinstance(prompt, mx.array):
            token_ids = prompt.tolist()
        elif isinstance(prompt, list):
            token_ids = prompt
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        token_ids = mx.array(token_ids)
      
        # add sequence id to the cache  
        # add sequence id to the cache  
        kv_cache = make_prompt_cache(self.model, max_kv_size=max_kv_size, block_size=block_size, num_blocks=num_blocks)
        batch = token_ids[None, :]
        # prefill the cache, no mask, we are justt doing a prefix only pass
        self.model(batch, cache=kv_cache)

        next_token_id = int(token_ids[-1])

        request_id = next(self._next_request_id_counter)

        # make the paged kv cache

        self.requests[request_id] = _RequestState(
            request_id=request_id,
            prompt_tokens=token_ids,
            detokenizer=self.tokenizer.detokenizer,
            max_output_tokens=max_tokens,
            kv_cache=kv_cache,
            next_input_token_id=next_token_id,

        )

        return request_id
    

    def step(self) -> List[AsyncResponse]:
        """
        Perform a single step of generation for all active requests.
        """
        
        
        # collect all the requests into one batch 
        active_requests = [r for r in self.requests.values() if not r.finished]
        if len(active_requests) == 0:
            return []

        # get the token ids
        next_ids = mx.array([r.next_input_token_id for r in active_requests], dtype=mx.uint32)
        batch = next_ids[:, None]  # shape (B, 1): add sequence-length dimension


        # 3) build a full boolean mask (causal + pad-mask) of shape [B, K, K]
        #    K = max over all offset lengths in this batch
        # These are the correct current lengths of each sequence in the batch
        # before processing the current token. These are what RoPE needs.
        current_batch_rope_offsets = mx.array([r.kv_cache[0].offset[0] for r in active_requests],
                   dtype=mx.int32)   # shape (B,)
        print(f"[continuous_batcher.py] batch.shape: {batch.shape}") # DEBUG
        print(f"[continuous_batcher.py] current_batch_rope_offsets.shape: {current_batch_rope_offsets.shape}") # DEBUG
        print(f"[continuous_batcher.py] current_batch_rope_offsets: {current_batch_rope_offsets}") # DEBUG
            
        # bottom-line API: pass lengths to get both causal+pad
        mask = create_attention_mask(
            h=batch, # Pass current batch to help determine query seq length (N_q)
            cache=None, # Mask creation doesn't need the actual cache object if lengths are given
            lengths=current_batch_rope_offsets, # Pass the true current lengths
            return_array=True

        )  # shape [B, N_q, K_max]

        # is this correct
        caches_for_model = active_requests[0].kv_cache # Still using req[0]'s cache for KV store

        logits = self.model(batch, mask=mask, cache=caches_for_model, rope_offsets=current_batch_rope_offsets)
        # logits shape is (B, 1, V)

        # compute log probs + sample (or greefy next token)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        logprobs = logprobs[:, 0, :]

            # sample next token
        sampler = self.sampler or (lambda x: mx.argmax(x, axis=-1))
        new_ids = sampler(logprobs)   # shape [B]


        # 7) build one AsyncResponse per request & advance its state
        responses: List[AsyncResponse] = []
        for i, req in enumerate(active_requests):
            tok = int(new_ids[i].item())
            req.next_input_token_id = tok
            req.tokens_generated += 1
            req.all_generated_token_ids.append(tok)

            # detokenize only the newly-added token
            req.detokenizer.add_token(tok)
            segment = req.detokenizer.last_segment

            # check stopping criteria
            done = False
            reason = None
            if tok in req.detokenizer._eos_token_ids:
                done = True
                reason = "eos"
            elif req.tokens_generated >= req.max_output_tokens:
                done = True
                reason = "length"

            if done:
                req.finished = True
                req.finish_reason = reason
                req.detokenizer.finalize()
                req.full_text = req.detokenizer.text

            responses.append(
                AsyncResponse(
                    request_id=req.request_id,
                    token_id=tok,
                    logprobs=logprobs[i],
                    text_segment=segment,
                    finished=done,
                    finish_reason=reason,
                )
            )

        return responses



    def get_full_text(self, request_id: int) -> Optional[str]:
        if request_id in self.requests:
            return self.requests[request_id].full_text
        return None

    def is_finished(self, request_id: int) -> bool:
        if request_id in self.requests:
            return self.requests[request_id].finished
        return True # If not found, assume finished/invalid

    def has_active_requests(self) -> bool:
        return any(not r.finished for r in self.requests.values())
