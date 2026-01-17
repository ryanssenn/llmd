from dataclasses import dataclass, fields
from typing import Optional
import json
from pathlib import Path
import gc
from tqdm import tqdm

from torch import nn
import torch
from safetensors.torch import safe_open


@dataclass
class MistralConfig:
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    tie_word_embeddings: bool
    torch_dtype: str
    vocab_size: int

    @classmethod
    def from_json(cls, path: str) -> "MistralConfig":
        with open(path) as f:
            data = json.load(f)

        allowed = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in allowed}

        return cls(**filtered)



class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        down_proj = self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MistralAttention(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self):
        pass



class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        pass


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MistralAttention(config=config, layer_idx=layer_idx)
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self):
        pass




class MistralRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  

    def __init__(self, config: MistralConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: MistralConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


    def forward(self, x, position_ids):
        pass


class MistralModel(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MistralRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def forward(self):
        pass


class MistralForCausalLM(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config, self.model = config, MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)


    @classmethod
    def from_pretrained(cls, path: str, config: MistralConfig, device, dtype):
        path = Path(path)
        files = sorted(path.glob("*.safetensors"))
        if not files: raise FileNotFoundError(f"No safetensors in {path}")
        
        # Create on meta device
        with torch.device('meta'): model = cls(config)
        
        weights = {}
        for file in tqdm(files, desc="Loading weights", unit="file", ncols=80):
            with safe_open(file, framework="pt", device=str(device)) as sf:
                for key in sf.keys():
                    tensor = sf.get_tensor(key)
                    weights[key] = tensor.to(dtype) if tensor.dtype in (torch.float16, torch.float32, torch.bfloat16) else tensor
        
        # Materialize weights from meta device
        def materialize(module, prefix=""):
            for name, child in module.named_children(): materialize(child, f"{prefix}{name}.")
            
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{prefix}{name}"
                if full_name in weights: module._parameters[name] = nn.Parameter(weights[full_name], requires_grad=False)
                elif param.device.type == 'meta': module._parameters[name] = nn.Parameter(torch.empty(param.shape, device=device, dtype=dtype).normal_(std=0.02), requires_grad=False)
            
        materialize(model)
        del weights; gc.collect(); torch.cuda.empty_cache()
        
        return model
    




