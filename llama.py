from __future__ import annotations

import math
import sys
import time
from typing import TypeVar, Generic, Optional

import cupy as cp

from config.config import ModelArgs
from tokenizer import Tokenizer
from attention import Attention
from RoPE import RoPE
from utils import load_parameters, FeedForward, RMSNorm

cp.set_printoptions(suppress=True)

class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args,
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps,
        )

    def __call__(
        self,
        x: cp.ndarray,
        start_pos: int,
        mask: Optional[cp.ndarray],
        freqs_cos: cp.ndarray,
        freqs_sin: cp.ndarray,
    ) -> cp.ndarray:
        norm_x = self.input_layernorm(x)
        h1 = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = cp.asarray(x) + h1

        norm_z = self.post_attention_layernorm(z)
        h2 = self.feed_forward(norm_z)
        out = z + h2

        return out


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args

        weight = load_parameters(model_path)
        self.tok_embedding: cp.ndarray = weight.get("model.embed_tokens.weight")

        # RoPE #1
        base = 10000
        head_dim = args.dim // args.n_heads
        inv_freq: cp.ndarray = 1.0 / (
            base ** (cp.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim)
        )
        t: cp.ndarray = cp.arange(args.max_seq_len)
        freqs: cp.ndarray = cp.outer(t, inv_freq)
        self.freqs_cos: cp.ndarray = cp.cos(freqs)
        self.freqs_sin: cp.ndarray = cp.sin(freqs)

        self.layers = [
            TransformerBlock(weight, layer_id, args)
            for layer_id in range(args.n_layers)
        ]

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        self.lm_head_weight: cp.ndarray = weight.get("lm_head.weight").T

        del weight

    def __call__(self, input_ids: cp.ndarray, start_pos: int) -> cp.ndarray:
        _, L = input_ids.shape
        h = self.tok_embedding[input_ids.get()]
        
        freqs_cos = self.freqs_cos[start_pos : start_pos + L]
        freqs_sin = self.freqs_sin[start_pos : start_pos + L]

        mask: Optional[cp.ndarray] = None
        if L > 1:
            mask = cp.full((L, L), float("-inf"))
            mask = cp.triu(mask, k=1)
            mask = cp.concatenate([cp.zeros((L, start_pos)), mask], axis=1)

        for layer in self.layers:
            h = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        h = self.norm(h)
        logit = cp.asarray(h[:, [-1], :]) @ cp.asarray(self.lm_head_weight)

        return logit

    def generate(
        self, input_ids: cp.ndarray, max_new_tokens: int
    ) -> Generator[cp.ndarray, None, None]:
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos
            logits = self(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id


if __name__ == "__main__":
    args = ModelArgs()

    tokenizer = Tokenizer(args.tokenizer_path)
    model = Llama(args.model_path, args)

    if len(sys.argv) == 1:
        prompt = "I have a dream"
    else:
        prompt = sys.argv[1]

    print(f"\n{prompt}", end="")
    input_ids = cp.array([tokenizer.encode(prompt)])
    start = time.time()
    _, L = input_ids.shape
    for id in model.generate(input_ids, args.max_new_tokens):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    print(
        f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s"
    )
