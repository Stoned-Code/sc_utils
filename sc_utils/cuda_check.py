import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}\n")

if torch.cuda.is_available():
    print(f"CUDA version (from PyTorch): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"   Memory Reserved:  {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"   Total Memory:     {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    # Quick functionality test
    try:
        device = torch.device("cuda")
        x = torch.randn(1024, 1024, device=device)
        y = torch.randn(1024, 1024, device=device)
        z = torch.matmul(x, y)
        print(f"\n✅ GPU tensor test passed! (Matmul on {device})")
    except Exception as e:
        print(f"\n❌ GPU tensor test failed: {e}")
        
else:
    print("❌ CUDA is NOT available")
    print("\nPossible issues:")
    print("- NVIDIA driver not installed or not loaded")
    print("- Wrong PyTorch CUDA wheel installed")
    print("- GPU not detected by nvidia-smi")
    
print(f"\nCurrent device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")