import torch
import platform

def get_torch_backend():
    """Returns the appropriate PyTorch backend based on the hardware platform."""
    if torch.cuda.is_available():
        if "AMD" in platform.system():
            import torch.backends.amd_miopen as backend
            print("Using MIOpen as the PyTorch backend.")
        else:
            import torch.backends.cudnn as backend
            print("Using cuDNN as the PyTorch backend.")
    else:
        if "AMD" in platform.system():
            import torch.backends.amdfft as backend
            torch.backends.amd_spblas.enabled = True
            print("Using AMDL as the PyTorch backend.")
        else:
            import torch.backends.mkl as backend
            torch.backends.mkl.enabled = True
            print("Using MKL as the PyTorch backend.")
    return backend

if __name__ == '__main__':
    # Example usage:
    backend = get_torch_backend()
    backend.benchmark = True  # Set benchmark flag to enable hardware-specific optimization
