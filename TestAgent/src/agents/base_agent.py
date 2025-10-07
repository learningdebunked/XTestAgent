"""
Base Agent Class

This module defines the abstract base class for all agents in the TestAgent framework.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic

T = TypeVar('T')

class BaseAgent(ABC, Generic[T]):
    """Abstract base class for all agents in the TestAgent framework."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent with optional configuration.
        
        Args:
            config: Optional configuration dictionary for the agent.
        """
        self.config = config or {}
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the agent's resources.
        
        This method should be called before any other methods.
        """
        self._initialized = True
        
    @abstractmethod
    async def execute(self, input_data: T, **kwargs) -> T:
        """Execute the agent's main functionality.
        
        Args:
            input_data: Input data for the agent to process.
            **kwargs: Additional keyword arguments for customization.
            
        Returns:
            The processed output data.
            
        Raises:
            RuntimeError: If the agent is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
            
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        self._initialized = False
        
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key to retrieve.
            default: Default value to return if key is not found.
            
        Returns:
            The configuration value or the default if not found.
        """
        return self.config.get(key, default)
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update the agent's configuration.
        
        Args:
            updates: Dictionary of configuration updates.
        """
        self.config.update(updates)
        
    @property
    def is_initialized(self) -> bool:
        """Check if the agent is initialized."""
        return self._initialized
    
    def __str__(self) -> str:
        """Return a string representation of the agent."""
        return f"{self.__class__.__name__}(initialized={self._initialized})"
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the agent."""
        return f"{self.__class__.__name__}(config={self.config}, initialized={self._initialized})"
