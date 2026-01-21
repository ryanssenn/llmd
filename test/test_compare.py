import sys
sys.path.insert(0, "..")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import MistralConfig, MistralForCausalLM


model_path = "../models/Ministral-3b-instruct"

def load_hf_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cpu"
    )
    model.eval()
    return model


def load_our_model():
    config = MistralConfig.from_json("../models/Ministral-3b-instruct/config.json")
    model = MistralForCausalLM.from_pretrained(
        model_path,
        config=config,
        device="cpu",
        dtype=torch.float16,
    )
    model.eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained(model_path)


def test_logits(prompt, num_tokens=10):
    tokenizer = load_tokenizer()
    hf_model = load_hf_model()
    our_model = load_our_model()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    generated = []

    with torch.no_grad():
        for i in range(num_tokens):
            hf_logits = hf_model(input_ids).logits
            our_logits = our_model(input_ids)

            max_diff = (hf_logits - our_logits).abs().max().item()

            hf_top10 = torch.topk(hf_logits[0, -1], 10).indices
            our_top10 = torch.topk(our_logits[0, -1], 10).indices
            top10_match = hf_top10.tolist() == our_top10.tolist()

            next_token = hf_top10[0].item()
            token_str = tokenizer.decode([next_token])
            generated.append(token_str)

            status = "✓" if top10_match else "✗"
            print(f"[{i+1}] {status} max_diff={max_diff:.6f} token={token_str!r}")

            if not top10_match:
                hf_top = torch.topk(hf_logits[0, -1], 10)
                our_top = torch.topk(our_logits[0, -1], 10)
                print("    HF:  ", end="")
                print(", ".join(f"{tokenizer.decode([t])}:{v:.2f}" for t, v in zip(hf_top.indices.tolist(), hf_top.values.tolist())))
                print("    Ours:", end="")
                print(", ".join(f"{tokenizer.decode([t])}:{v:.2f}" for t, v in zip(our_top.indices.tolist(), our_top.values.tolist())))
                break

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)


if __name__ == "__main__":
    test_logits("Hello, how are you doing today?", num_tokens=5)
