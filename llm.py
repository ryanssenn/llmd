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


class LLM:
    def __init__(self, model_path: str, device: str = "cpu", dtype=torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype

        config = MistralConfig.from_json(f"{model_path}/config.json")
        self.model = MistralForCausalLM.from_pretrained(model_path, config, device, dtype)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self._request_queue: queue.Queue[Request] = queue.Queue()
        self._loop_running = False
        self._loop_thread = None

        print(f"Engine ready | device={device}")

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
            # Drain queue
            while True:
                try:
                    req = self._request_queue.get_nowait()
                    active.append(req)
                except queue.Empty:
                    break

            if not active:
                if not self._loop_running:
                    break
                time.sleep(0.001)
                continue

            # Process one step for each active request
            # TODO: batch requests together for efficiency
            completed = []
            t0 = time.perf_counter()
            for req in active:
                # Build input: prompt + generated so far
                input_ids = torch.tensor([req.token_ids + req.generated], device=self.device)

                # Forward pass
                logits = self.model(input_ids)
                next_logits = logits[0, -1]

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
                active.remove(req)

        print()  # Newline after tok/s output


