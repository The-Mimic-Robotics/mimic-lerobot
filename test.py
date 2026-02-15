import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version used by PyTorch: {torch.version.cuda}")
print(f"Is CUDA available? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # This is the hardware architecture of your actual GPU
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU Compute Capability: {capability[0]}.{capability[1]}")
    
    # This is what PyTorch supports
    print(f"PyTorch Architecture List: {torch.cuda.get_arch_list()}")
else:
    print("CUDA is not available to PyTorch.")