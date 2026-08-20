import sys
import platform

def generate_environment_report():
    try:
        import torch
        import transformers
        import numpy
        import pandas
        import sklearn
        
        report = f"""# Environment Report

**Date/Time of record:** 2026-08-18

This document establishes the hardware and software configuration used during the development of this project.

## Hardware
- **OS:** {platform.system()} {platform.release()}
- **GPU Model:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}
- **VRAM Total:** {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 'N/A'} GB
- **NVIDIA Driver:** 592.82 (Reported by nvidia-smi)
- **CUDA Device Count:** {torch.cuda.device_count() if torch.cuda.is_available() else 0}

## Software Versions
- **Python:** {sys.version.split(' ')[0]}
- **PyTorch:** {torch.__version__}
- **Transformers:** {transformers.__version__}
- **NumPy:** {numpy.__version__}
- **Pandas:** {pandas.__version__}
- **scikit-learn:** {sklearn.__version__}
- **PyTorch CUDA Runtime:** {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
- **Project Environment Path:** .venv
"""
        
        with open('docs/environment_report.md', 'w') as f:
            f.write(report)
        print("Environment report generated successfully.")
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == '__main__':
    generate_environment_report()
