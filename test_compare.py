import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import MistralConfig, MistralForCausalLM


def load_hf_model():
    model = AutoModelForCausalLM.from_pretrained(
        "models/Ministral-3b-instruct",
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    model.eval()
    return model


def load_our_model():
    config = MistralConfig.from_json("models/Ministral-3b-instruct/config.json")
    model = MistralForCausalLM.from_pretrained(
        "models/Ministral-3b-instruct/",
        config=config,
        device="cpu",
        dtype=torch.float16,
    )
    model.eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained("models/Ministral-3b-instruct")


def test_logits(prompt):
    tokenizer = load_tokenizer()

    print("Loading HF model...")
    hf_model = load_hf_model()

    print("Loading our model...")
    our_model = load_our_model()

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"\nPrompt: {prompt!r}")
    print(f"Token IDs: {input_ids.tolist()}")

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        our_logits = our_model(input_ids)

    print(f"\nHF logits shape: {hf_logits.shape}")
    print(f"Our logits shape: {our_logits.shape}")

    # Compare
    max_diff = (hf_logits - our_logits).abs().max().item()
    mean_diff = (hf_logits - our_logits).abs().mean().item()
    close = torch.allclose(hf_logits, our_logits, atol=1e-2, rtol=1e-2)

    print(f"\nMax diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Close (atol=1e-2): {close}")

    # Show top 5 predictions from each
    print(f"\nHF top 5 (last token):")
    top5_hf = torch.topk(hf_logits[0, -1], 5)
    for val, idx in zip(top5_hf.values, top5_hf.indices):
        print(f"  {tokenizer.decode([idx])!r}: {val.item():.4f}")

    print(f"\nOur top 5 (last token):")
    top5_ours = torch.topk(our_logits[0, -1], 5)
    for val, idx in zip(top5_ours.values, top5_ours.indices):
        print(f"  {tokenizer.decode([idx])!r}: {val.item():.4f}")


if __name__ == "__main__":
    test_logits("Hello, how are you doing today?")
