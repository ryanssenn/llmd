from model import MistralConfig, MistralForCausalLM
import torch


config = MistralConfig.from_json("models/Ministral-3b-instruct/config.json")

model = MistralForCausalLM.from_pretrained(
    "models/Ministral-3b-instruct/",
    config=config,
    device="cpu",
    dtype=torch.float16,
)

