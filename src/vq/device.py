import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_torch_dtype(device: torch.device) -> torch.dtype:
    """Return appropriate dtype. Use float16 only on CUDA."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32
