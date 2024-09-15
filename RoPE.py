import cupy as cp

cp.set_printoptions(suppress=True)

def RoPE(
    xq: cp.ndarray,
    xk: cp.ndarray,
    freqs_cos: cp.ndarray,
    freqs_sin: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray]:
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))

    xq_r, xq_i = cp.split(xqri, 2, axis=-1)
    xk_r, xk_i = cp.split(xkri, 2, axis=-1)

    xq_r = xq_r.squeeze(-1)
    xq_i = xq_i.squeeze(-1)
    xk_r = xk_r.squeeze(-1)
    xk_i = xk_i.squeeze(-1)

    freqs_cos = cp.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = cp.expand_dims(freqs_sin, axis=(0, 2))

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    xq_out = cp.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out = cp.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    xk_out = xk_out.reshape(xk_out.shape[:-2] + (-1,))

    return xq_out, xk_out

