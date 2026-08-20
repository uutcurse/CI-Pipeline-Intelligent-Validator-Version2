import uvicorn
import yaml
import sys
from pathlib import Path

# Ensure src is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    with open('configs/api.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    uvicorn.run(
        "src.api.main:app", 
        host=config['server']['host'], 
        port=config['server']['port'],
        log_level=config['server']['log_level'],
        reload=True,
        app_dir=str(Path(__file__).resolve().parent.parent)
    )
