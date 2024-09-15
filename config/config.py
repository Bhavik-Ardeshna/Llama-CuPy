from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    model_path: str             = "./model/llama.model.npz"
    tokenizer_path: str         = "./model/tokenizer.model.np"
    dim: int                    = 288       # D
    n_layers: int               = 6
    n_heads: int                = 6         # QHN, HN, HD = 48
    n_kv_heads: Optional[int]   = None      # KVHN = 6
    vocab_size: int             = 32000     # VS
    max_seq_len: int            = 1024       # M
    max_new_tokens: int         = 512
    norm_eps: float             = 1e-6
    max_batch_size: int         = 1