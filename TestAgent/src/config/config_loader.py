"""
Configuration Loader for TestAgentX

Loads and manages configuration from YAML files, environment variables,
and command-line arguments. Eliminates hardcoded constants.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Main configuration class for TestAgentX"""
    
    # Code Encoder
    code_encoder: Dict[str, Any] = field(default_factory=dict)
    
    # Test Generation
    test_generation: Dict[str, Any] = field(default_factory=dict)
    
    # Fuzzy Validation
    fuzzy_validation: Dict[str, Any] = field(default_factory=dict)
    
    # Patch Verification
    patch_verification: Dict[str, Any] = field(default_factory=dict)
    
    # Knowledge Graph
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    
    # Mutation Testing
    mutation_testing: Dict[str, Any] = field(default_factory=dict)
    
    # Chaos Engineering
    chaos_engineering: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation
    evaluation: Dict[str, Any] = field(default_factory=dict)
    
    # Logging
    logging: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    paths: Dict[str, Any] = field(default_factory=dict)
    
    # Feature Flags
    features: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Path to config value (e.g., 'test_generation.llm.temperature')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.__dict__
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation.
        
        Args:
            key_path: Path to config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.__dict__
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value


class ConfigLoader:
    """Loads configuration from multiple sources with priority:
    1. Command-line arguments (highest priority)
    2. Environment variables
    3. User config file
    4. Default config file (lowest priority)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.
        
        Args:
            config_path: Path to user configuration file
        """
        self.config_path = config_path
        self.default_config_path = self._find_default_config()
        
    def _find_default_config(self) -> Path:
        """Find the default configuration file"""
        # Try multiple locations
        possible_paths = [
            Path(__file__).parent.parent.parent / 'config' / 'default_config.yaml',
            Path.cwd() / 'config' / 'default_config.yaml',
            Path.home() / '.testagentx' / 'config.yaml'
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        logger.warning("Default config file not found, using empty config")
        return None
    
    def load(self) -> Config:
        """Load configuration from all sources.
        
        Returns:
            Merged Config object
        """
        config = Config()
        
        # 1. Load default config
        if self.default_config_path and self.default_config_path.exists():
            default_data = self._load_yaml(self.default_config_path)
            self._merge_config(config, default_data)
            logger.info(f"Loaded default config from {self.default_config_path}")
        
        # 2. Load user config (if provided)
        if self.config_path:
            user_path = Path(self.config_path)
            if user_path.exists():
                user_data = self._load_yaml(user_path)
                self._merge_config(config, user_data)
                logger.info(f"Loaded user config from {user_path}")
            else:
                logger.warning(f"User config file not found: {user_path}")
        
        # 3. Override with environment variables
        self._load_from_env(config)
        
        # 4. Validate configuration
        self._validate_config(config)
        
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Error loading config from {path}: {e}")
            return {}
    
    def _merge_config(self, config: Config, data: Dict[str, Any]) -> None:
        """Merge configuration data into Config object"""
        for key, value in data.items():
            if hasattr(config, key):
                if isinstance(value, dict) and isinstance(getattr(config, key), dict):
                    # Merge dictionaries
                    getattr(config, key).update(value)
                else:
                    setattr(config, key, value)
    
    def _load_from_env(self, config: Config) -> None:
        """Load configuration from environment variables.
        
        Environment variables should be prefixed with TESTAGENTX_
        and use double underscores for nesting.
        
        Example: TESTAGENTX_TEST_GENERATION__LLM__TEMPERATURE=0.8
        """
        prefix = "TESTAGENTX_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower().replace('__', '.')
                
                # Try to parse value
                parsed_value = self._parse_env_value(value)
                
                # Set in config
                config.set(config_key, parsed_value)
                logger.debug(f"Set config from env: {config_key} = {parsed_value}")
    
    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Try boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Try int
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _validate_config(self, config: Config) -> None:
        """Validate configuration values"""
        # Validate embedding dimensions
        if config.code_encoder.get('embedding_dim', 0) <= 0:
            logger.warning("Invalid embedding_dim in code_encoder, using default 768")
            config.code_encoder['embedding_dim'] = 768
        
        # Validate thresholds
        threshold = config.fuzzy_validation.get('threshold', 0.7)
        if not 0 <= threshold <= 1:
            logger.warning(f"Invalid threshold {threshold}, using default 0.7")
            config.fuzzy_validation['threshold'] = 0.7
        
        # Validate RL parameters
        gamma = config.test_generation.get('rl_prioritization', {}).get('gamma', 0.99)
        if not 0 <= gamma <= 1:
            logger.warning(f"Invalid gamma {gamma}, using default 0.99")
            config.test_generation.setdefault('rl_prioritization', {})['gamma'] = 0.99
        
        # Validate paths exist or create them
        for path_key, path_value in config.paths.items():
            path = Path(path_value)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
    
    def save(self, config: Config, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Config object to save
            output_path: Path to save configuration
        """
        try:
            config_dict = {
                'code_encoder': config.code_encoder,
                'test_generation': config.test_generation,
                'fuzzy_validation': config.fuzzy_validation,
                'patch_verification': config.patch_verification,
                'knowledge_graph': config.knowledge_graph,
                'mutation_testing': config.mutation_testing,
                'chaos_engineering': config.chaos_engineering,
                'evaluation': config.evaluation,
                'logging': config.logging,
                'paths': config.paths,
                'features': config.features
            }
            
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise


# Global configuration instance
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance.
    
    Returns:
        Global Config object
    """
    global _global_config
    
    if _global_config is None:
        loader = ConfigLoader()
        _global_config = loader.load()
    
    return _global_config


def set_config(config: Config) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Config object to set as global
    """
    global _global_config
    _global_config = config


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Reloaded Config object
    """
    loader = ConfigLoader(config_path)
    config = loader.load()
    set_config(config)
    return config


if __name__ == "__main__":
    # Example usage
    print("Loading TestAgentX configuration...")
    
    loader = ConfigLoader()
    config = loader.load()
    
    print("\nConfiguration loaded:")
    print(f"  Code encoder embedding dim: {config.get('code_encoder.embedding_dim')}")
    print(f"  LLM temperature: {config.get('test_generation.llm.temperature')}")
    print(f"  Fuzzy validation threshold: {config.get('fuzzy_validation.threshold')}")
    print(f"  RL gamma: {config.get('test_generation.rl_prioritization.gamma')}")
    print(f"  Neo4j URI: {config.get('knowledge_graph.neo4j.uri')}")
    
    # Save example
    # loader.save(config, 'config/my_config.yaml')
