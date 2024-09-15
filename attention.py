import math
import cupy as cp

from typing import Optional

from RoPE import RoPE
from utils import repeat_kv, softmax
from config.config import ModelArgs

cp.set_printoptions(suppress=True)

class Attention:
    def __init__(
        self,
        q_weight: cp.ndarray,
        k_weight: cp.ndarray,
        v_weight: cp.ndarray,
        o_weight: cp.ndarray,
        args: ModelArgs,
    ):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        self.cache_k = cp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = cp.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def __call__(
        self,
        x: cp.ndarray,
        start_pos: int,
        mask: Optional[cp.ndarray],
        freqs_cos: cp.ndarray,
        freqs_sin: cp.ndarray,
    ) -> cp.ndarray:
        B, L, _ = x.shape

        # QKV
        xq = cp.asarray(x) @ cp.asarray(self.q_weight)
        xk = cp.asarray(x) @ cp.asarray(self.k_weight)
        xv = cp.asarray(x) @ cp.asarray(self.v_weight)

        xq = xq.reshape(B, L, self.n_local_heads, self.head_dim)
        xk = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        # RoPE
        xq, xk = RoPE(xq, xk, freqs_cos, freqs_sin)

        # KV Cache
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        ks = self.cache_k[:B, : start_pos + L]
        vs = self.cache_v[:B, : start_pos + L]

        # GQA
        xk = repeat_kv(ks, self.n_rep)
        xv = repeat_kv(vs, self.n_rep)

        xq = xq.transpose(0, 2, 1, 3)
        xk = xk.transpose(0, 2, 1, 3)
        xv = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        attention = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        output = attention @ xv

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output = cp.asarray(output) @ cp.asarray(self.o_weight)

        return output
