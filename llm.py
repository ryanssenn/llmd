#!/usr/bin/env python3
"""Minimal LLM inference engine with async request queue."""
import threading
import queue
import time
import concurrent.futures
from collections import deque
from dataclasses import dataclass, field

import torch
from transformers import AutoTokenizer

from model import MistralConfig, MistralForCausalLM


@dataclass
class Request:
    """Internal request tracking."""
    token_ids: list[int]
    max_tokens: int
    temperature: float
    future: concurrent.futures.Future
    generated: list[int] = field(default_factory=list)
    slot_idx: int = -1
    cache_position: int = 0


class LLM:
    def __init__(self, model_path: str, device: str = "cpu", dtype=torch.float16, num_slots: int = 4, max_seq_len: int = 4096):
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len

        config = MistralConfig.from_json(f"{model_path}/config.json")
        self.model = MistralForCausalLM.from_pretrained(model_path, config, device, dtype)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Allocate KV cache for each attention layer
        head_dim = config.hidden_size // config.num_attention_heads
        for layer in self.model.model.layers:
            attn = layer.self_attn
            attn.k_cache = torch.zeros(num_slots, config.num_key_value_heads, max_seq_len, head_dim, device=self.device, dtype=dtype)
            attn.v_cache = torch.zeros(num_slots, config.num_key_value_heads, max_seq_len, head_dim, device=self.device, dtype=dtype)

        self._free_slots: list[int] = list(range(num_slots))
        self._request_queue: queue.Queue[Request] = queue.Queue()
        self._loop_running = False
        self._loop_thread = None

        print(f"Engine ready | device={device} | slots={num_slots} | max_seq_len={max_seq_len}")

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> concurrent.futures.Future:
        """Submit a prompt for generation. Returns a Future."""
        if not self._loop_running:
            self._loop_running = True
            self._loop_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._loop_thread.start()

        future = concurrent.futures.Future()
        token_ids = self.tokenizer.encode(prompt)
        request = Request(token_ids=token_ids, max_tokens=max_tokens, temperature=temperature, future=future)
        self._request_queue.put(request)
        return future

    def stop(self):
        """Stop the background inference loop."""
        self._loop_running = False
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        """Sample next token from logits."""
        if temperature == 0:
            return logits.argmax(dim=-1).item()
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).item()

    @torch.inference_mode()
    def _inference_loop(self):
        """Background loop that processes requests."""
        active: list[Request] = []
        decode_times: deque[tuple[int, float]] = deque(maxlen=16)

        while self._loop_running or active:
            # Drain queue, assign slots to new requests
            while self._free_slots:
                try:
                    req = self._request_queue.get_nowait()
                    req.slot_idx = self._free_slots.pop()
                    req.cache_position = 0
                    active.append(req)
                except queue.Empty:
                    break

            if not active:
                if not self._loop_running:
                    break
                time.sleep(0.001)
                continue

            # Process one step for each active request
            completed = []
            t0 = time.perf_counter()
            for req in active:
                # First step: prefill with prompt, subsequent steps: single token
                if req.cache_position == 0:
                    input_ids = torch.tensor([req.token_ids], device=self.device)
                else:
                    input_ids = torch.tensor([[req.generated[-1]]], device=self.device)

                # Forward pass with cache
                logits = self.model(input_ids, slot_idx=req.slot_idx, cache_position=req.cache_position)
                next_logits = logits[0, -1]

                # Update cache position
                req.cache_position += input_ids.shape[1]

                # Sample
                next_token = self._sample(next_logits, req.temperature)
                req.generated.append(next_token)

                # Check completion
                if len(req.generated) >= req.max_tokens or next_token == self.tokenizer.eos_token_id:
                    output_text = self.tokenizer.decode(req.generated, skip_special_tokens=True)
                    req.future.set_result(output_text)
                    completed.append(req)

            # Track timing
            decode_times.append((len(active), time.perf_counter() - t0))
            tok_per_sec = sum(n for n, _ in decode_times) / max(sum(t for _, t in decode_times), 1e-9)
            print(f"\rtok/s={tok_per_sec:.1f}", end="", flush=True)

            for req in completed:
                self._free_slots.append(req.slot_idx)
                active.remove(req)

        print()  # Newline after tok/s output


