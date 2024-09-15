import numpy as np
import cupy as cp

cp.set_printoptions(suppress=True)

def load_parameters(model_path):
    return np.load(model_path)

def softmax(x: cp.ndarray) -> cp.ndarray:
    exp_x = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
    return exp_x / cp.sum(exp_x, axis=-1, keepdims=True)

def silu(x: cp.ndarray) -> cp.ndarray:
    return x * (1 / (1 + cp.exp(-x)))

def repeat_kv(x: cp.ndarray, n_rep: int) -> cp.ndarray:
    if n_rep == 1:
        return x
    return cp.repeat(x, n_rep, axis=2)

class RMSNorm:
    def __init__(self, weight: cp.ndarray, eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        z = (x**2).mean(-1, keepdims=True) + self.eps
        z = cp.asarray(x) / cp.sqrt(cp.asarray(z))
        return cp.asarray(z) * cp.asarray(self.weight)

class FeedForward:
    def __init__(
        self,
        up_weight: cp.ndarray,
        gate_weight: cp.ndarray,
        down_weight: cp.ndarray
    ):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def __call__(self, x: cp.ndarray) -> cp.ndarray:
        swish = silu(cp.asarray(x) @ cp.asarray(self.gate_weight))
        x_V = cp.asarray(x) @ cp.asarray(self.up_weight)
        x = swish * x_V
        x = cp.asarray(x) @ cp.asarray(self.down_weight)
        return x
        