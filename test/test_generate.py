#!/usr/bin/env python3
"""Test script for LLM inference with multiple prompts."""
import sys
sys.path.insert(0, "..")

from llm import LLM

prompts = [
    "Calculate 5 plus 3",
    "What color is the sky?",
    "Who painted the Sistine Chapel ceiling?",
    "What is the capital of France?",
    "How many days are in a year?"
]


def main():
    llm = LLM("../models/Ministral-3b-instruct")

    # Submit all prompts
    futures = []
    for prompt in prompts:
        future = llm.generate(prompt, max_tokens=100)
        futures.append((prompt, future))

    # Wait for and display results
    for prompt, future in futures:
        result = future.result()
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"Response: {result}")

    llm.stop()


if __name__ == "__main__":
    main()
