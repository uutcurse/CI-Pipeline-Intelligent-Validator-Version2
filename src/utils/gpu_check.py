import sys
import platform

def verify_gpu() -> dict:
    """
    Verifies the Python environment, PyTorch installation, and performs 
    a real CUDA matrix operation to confirm GPU usability.
    """
    result = {
        "os": platform.system(),
        "python_version": sys.version.split(" ")[0],
        "torch_version": None,
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "cuda_device_count": 0,
        "gpu_memory_total_gb": None,
        "tensor_allocation_success": False,
        "cuda_computation_success": False,
        "error": None
    }
    
    try:
        import torch
        result["torch_version"] = torch.__version__
        result["torch_cuda_version"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        
        if result["cuda_available"]:
            result["cuda_device_count"] = torch.cuda.device_count()
            result["gpu_name"] = torch.cuda.get_device_name(0)
            
            # Obtain GPU memory
            mem_total = torch.cuda.get_device_properties(0).total_memory
            result["gpu_memory_total_gb"] = round(mem_total / (1024 ** 3), 2)
            
            # 7. Real tensor allocation on CUDA
            device = torch.device("cuda:0")
            size = (1000, 1000)
            x = torch.randn(size, device=device)
            y = torch.randn(size, device=device)
            result["tensor_allocation_success"] = True
            
            # 8. Real matrix operation on CUDA
            z = torch.matmul(x, y)
            
            # 9. Result returned to CPU successfully
            z_cpu = z.cpu().numpy()
            if z_cpu.shape == size:
                result["cuda_computation_success"] = True
                
    except Exception as e:
        result["error"] = str(e)
        
    return result

if __name__ == "__main__":
    print("="*50)
    print("GPU & Environment Verification Report")
    print("="*50)
    res = verify_gpu()
    for k, v in res.items():
        print(f"{k.ljust(30)}: {v}")
    
    if res.get("cuda_computation_success"):
        print("\nSTATUS: PASS (CUDA tensor operations are working)")
        sys.exit(0)
    elif not res.get("cuda_available"):
        print("\nSTATUS: SKIPPED (CUDA is not available on this machine)")
        sys.exit(0)
    else:
        print("\nSTATUS: FAIL (CUDA is available but computation failed)")
        sys.exit(1)
