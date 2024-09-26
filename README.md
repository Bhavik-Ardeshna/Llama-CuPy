# GPU Accelerated Llama using CuPY.

- **CuPy Integration**: Utilizes CuPy for GPU-accelerated matrix operations, improving efficiency over traditional CPU-based PyTorch implementations.
- **Optimized for NVIDIA GPUs**: Built to take full advantage of NVIDIA CUDA architecture, reducing inference and training times.

## Prerequisites

To run **Llama-CuPy**, you'll need the following:

- **Python 3.8+**
- **CUDA Toolkit 11.2+**
- **CuPy** (`cupy-cuda11x`)
- **PyTorch** (`torch` with CUDA support)
- **NVIDIA GPU** (recommended for optimal performance)

### Installing Dependencies

1. Install PyTorch with GPU support:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install CuPy:

   ```bash
   pip install cupy-cuda11x
   ```
3. Clone the repository:

   ```bash
   git clone https://github.com/Bhavik-Ardeshna/Llama-CuPy.git
   cd Llama-CuPy
   ```

## Getting Started

### Running Inference

To run inference using a pre-trained LLaMA model, follow the steps below:

1. Load the model:

   ```python
   from llama_cupy import LlamaModel
   model = LlamaModel(model_name="llama-7b")
   ```
2. Generate text:

   ```python
   prompt = "Once upon a time in a faraway land,"
   generated_text = model.generate(prompt)
   print(generated_text)
   ```

## Configuration

The configuration files are stored in the `config` folder. You can modify model parameters, batch size, and training epochs by editing the YAML configuration files.

Example configuration (`config/default.yaml`):

```yaml
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
```

## Benchmarks

**Llama-CuPy** achieves significant speed-ups on large models compared to the standard CPU-based implementation. Here are some benchmark results on an NVIDIA A100 GPU:

| Model | Inference Time | Speed-up vs CPU |
| ----- | -------------- | --------------- |
| LLaMA | 12ms           | 2.5x            |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- This repository is inspired by the [LLaMA](https://github.com/facebookresearch/llama) model by FAIR.
- Thanks to the [CuPy](https://github.com/cupy/cupy) team for their amazing library!
