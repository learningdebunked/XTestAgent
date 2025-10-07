"""
Agent Orchestrator

This module provides a class for coordinating multiple agents in the TestAgent framework.
"""
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic
from dataclasses import dataclass, field
import asyncio
import logging

from .base_agent import BaseAgent

T = TypeVar('T')

@dataclass
class AgentConfig:
    """Configuration for an agent in the orchestrator."""
    agent_class: Type[BaseAgent]
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True

class AgentOrchestrator(Generic[T]):
    """Orchestrates multiple agents to work together."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the orchestrator.
        
        Args:
            config: Configuration for the orchestrator.
        """
        self.config = config or {}
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.logger = logging.getLogger(__name__)
        self._initialized = False
        
    async def add_agent(self, name: str, agent_config: AgentConfig) -> None:
        """Add an agent to the orchestrator.
        
        Args:
            name: Unique name for the agent.
            agent_config: Configuration for the agent.
            
        Raises:
            ValueError: If an agent with the same name already exists.
        """
        if name in self.agent_configs:
            raise ValueError(f"Agent with name '{name}' already exists.")
            
        self.agent_configs[name] = agent_config
        
        if agent_config.enabled:
            agent = agent_config.agent_class(agent_config.config)
            self.agents[name] = agent
            
    async def initialize(self) -> None:
        """Initialize all registered agents in dependency order."""
        if self._initialized:
            return
            
        # Initialize agents in dependency order
        initialized = set()
        agents_to_init = set(self.agents.keys())
        
        while agents_to_init:
            made_progress = False
            
            for agent_name in list(agents_to_init):
                deps = self.agent_configs[agent_name].dependencies
                if all(dep in initialized for dep in deps):
                    self.logger.info("Initializing agent: %s", agent_name)
                    await self.agents[agent_name].initialize()
                    initialized.add(agent_name)
                    agents_to_init.remove(agent_name)
                    made_progress = True
                    
            if not made_progress and agents_to_init:
                # Circular dependency detected
                raise RuntimeError(
                    f"Circular dependency detected among agents: {agents_to_init}"
                )
                
        self._initialized = True
        self.logger.info("All agents initialized successfully.")
        
    async def execute_workflow(
        self,
        start_agent: str,
        input_data: T,
        max_steps: int = 100,
        **kwargs
    ) -> T:
        """Execute a workflow starting from the specified agent.
        
        Args:
            start_agent: Name of the agent to start the workflow from.
            input_data: Initial input data for the workflow.
            max_steps: Maximum number of steps to execute (prevents infinite loops).
            **kwargs: Additional keyword arguments for the workflow.
            
        Returns:
            The final output data from the workflow.
            
        Raises:
            ValueError: If the start agent is not found.
            RuntimeError: If the maximum number of steps is exceeded.
        """
        if start_agent not in self.agents:
            raise ValueError(f"Agent '{start_agent}' not found.")
            
        current_agent = start_agent
        current_data = input_data
        steps = 0
        
        while current_agent and steps < max_steps:
            self.logger.debug("Executing agent: %s", current_agent)
            agent = self.agents[current_agent]
            
            try:
                result = await agent.execute(current_data, **kwargs)
                
                # Determine next agent based on result or configuration
                # This is a simplified version - in practice, you might have more complex routing logic
                next_agent = self._determine_next_agent(current_agent, result)
                
                if next_agent is None:
                    return result
                    
                current_agent = next_agent
                current_data = result
                steps += 1
                
            except Exception as e:
                self.logger.error(
                    "Error in agent '%s': %s", current_agent, str(e), exc_info=True
                )
                raise
                
        if steps >= max_steps:
            raise RuntimeError(f"Maximum number of steps ({max_steps}) exceeded.")
            
        return current_data
    
    def _determine_next_agent(self, current_agent: str, result: Any) -> Optional[str]:
        """Determine the next agent to execute based on the current agent and result.
        
        This is a simplified implementation. In practice, you might have more complex
        routing logic based on the result or other factors.
        
        Args:
            current_agent: Name of the current agent.
            result: Result from the current agent.
            
        Returns:
            Name of the next agent to execute, or None if the workflow should end.
        """
        # This is a placeholder implementation
        # In a real implementation, you might have a more sophisticated routing system
        # that considers the result, agent configurations, and workflow definitions
        return None
    
    async def cleanup(self) -> None:
        """Clean up all agents in reverse initialization order."""
        for agent_name, agent in reversed(self.agents.items()):
            try:
                self.logger.info("Cleaning up agent: %s", agent_name)
                await agent.cleanup()
            except Exception as e:
                self.logger.error(
                    "Error cleaning up agent '%s': %s", agent_name, str(e), exc_info=True
                )
                
        self.agents.clear()
        self._initialized = False
        
    def get_agent(self, name: str) -> BaseAgent:
        """Get an agent by name.
        
        Args:
            name: Name of the agent to retrieve.
            
        Returns:
            The agent instance.
            
        Raises:
            KeyError: If the agent is not found.
        """
        return self.agents[name]
    
    def __contains__(self, name: str) -> bool:
        """Check if an agent with the given name exists."""
        return name in self.agents
    
    def __getitem__(self, name: str) -> BaseAgent:
        """Get an agent by name using dictionary syntax."""
        return self.get_agent(name)
    
    def __len__(self) -> int:
        """Get the number of agents in the orchestrator."""
        return len(self.agents)
    
    def __iter__(self):
        """Iterate over agent names."""
        return iter(self.agents)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
