import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional

def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # Assuming this file is at src/utils/config.py
    current_file = Path(__file__).resolve()
    # Go up 3 levels: src -> utils -> config.py
    return current_file.parent.parent.parent

def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

class ProjectPaths:
    """Helper class to resolve and provide absolute paths across the project."""
    
    def __init__(self, paths_config: Optional[Dict[str, Any]] = None):
        self.root = get_project_root()
        
        if paths_config is None:
            config_file = self.root / "configs" / "paths.yaml"
            if config_file.exists():
                paths_config = load_yaml_config(config_file)
            else:
                paths_config = {}
                
        # Resolve paths dynamically
        self.data_root = self.root / paths_config.get("data_root", "data")
        self.raw_data_root = self.data_root / paths_config.get("raw_data", "raw")
        self.intermediate_data_root = self.data_root / paths_config.get("intermediate_data", "intermediate")
        self.processed_data_root = self.data_root / paths_config.get("processed_data", "processed")
        self.splits_root = self.data_root / paths_config.get("splits_data", "splits")
        self.manifests_root = self.data_root / paths_config.get("manifests_data", "manifests")
        
        self.checkpoint_root = self.root / paths_config.get("checkpoint_root", "checkpoints")
        self.log_root = self.root / paths_config.get("log_root", "logs")
        
    def ensure_directories(self) -> None:
        """Creates all necessary directories if they don't exist."""
        dirs_to_create = [
            self.data_root, self.raw_data_root, self.intermediate_data_root, 
            self.processed_data_root, self.splits_root, self.manifests_root,
            self.checkpoint_root, self.log_root
        ]
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)

class ProjectConfig:
    """Main configuration loader including reproducibility settings."""
    
    def __init__(self):
        self.root = get_project_root()
        self.paths = ProjectPaths()
        
        # Load reproducibility and base settings
        base_config_path = self.root / "configs" / "base.yaml"
        self.settings = load_yaml_config(base_config_path) if base_config_path.exists() else {}
        
    @property
    def reproducibility(self) -> Dict[str, Any]:
        """Returns reproducibility settings like seed, precision, etc."""
        return self.settings.get("reproducibility", {})
