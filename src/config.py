"""
Configuration management for WIUT FinTech Hackathon
Handles loading and accessing configuration settings
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for the project"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _validate_config(self):
        """Validate required configuration sections"""
        required_sections = ['project', 'paths', 'data', 'model']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        logger.info("Configuration validation passed")
    
    def get(self, *keys, default=None) -> Any:
        """Get configuration value using dot notation"""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def get_paths(self) -> Dict[str, str]:
        """Get all file paths"""
        return self.config.get('paths', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.config.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config.get('features', {})
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration"""
        return self.config.get('validation', {})
    
    def get_submission_config(self) -> Dict[str, Any]:
        """Get submission configuration"""
        return self.config.get('submission', {})
    
    def get_random_seed(self) -> int:
        """Get random seed for reproducibility"""
        return self.config.get('reproducibility', {}).get('random_seed', 42)
    
    def get_team_id(self) -> str:
        """Get team ID"""
        return self.config.get('team', {}).get('id', 'TEAM_ID')
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        paths = self.get_paths()
        
        # Create directories from paths
        for key, path in paths.items():
            if key not in ['train_signals', 'test_signals', 'train_transactions', 
                          'test_transactions', 'sample_submission']:
                Path(path).mkdir(parents=True, exist_ok=True)
        
        # Create additional directories
        Path(self.config.get('paths', {}).get('artifacts', 'artifacts')).mkdir(parents=True, exist_ok=True)
        Path(self.config.get('paths', {}).get('models', 'artifacts/models')).mkdir(parents=True, exist_ok=True)
        Path(self.config.get('paths', {}).get('predictions', 'artifacts/predictions')).mkdir(parents=True, exist_ok=True)
        
        logger.info("All required directories ensured")


# Global configuration instance
_config_instance = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reset_config():
    """Reset global configuration instance (useful for testing)"""
    global _config_instance
    _config_instance = None