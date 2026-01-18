from model import MistralConfig, MistralForCausalLM
import torch


"""
config = MistralConfig.from_json("models/Ministral-3b-instruct/config.json")

model = MistralForCausalLM.from_pretrained(
    "models/Ministral-3b-instruct/",
    config=config,
    device="cpu",
    dtype=torch.float16,
)
"""

"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/Ministral-3b-instruct",
    dtype=torch.float16,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("models/Ministral-3b-instruct")

prompt = "Hello, how are you?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
"""