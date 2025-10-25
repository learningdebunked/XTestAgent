"""
Chaos Engineering Module for TestAgentX

Implements chaos engineering techniques to improve software dependability
as described in Section 3.6 of the paper.

This module provides:
- Fault injection
- Network chaos
- Resource exhaustion
- State mutation
- Time manipulation
"""

from .chaos_orchestrator import ChaosOrchestrator
from .fault_injector import FaultInjector
from .chaos_scenarios import ChaosScenario, ChaosScenarioType
from .resilience_tester import ResilienceTester

__all__ = [
    'ChaosOrchestrator',
    'FaultInjector',
    'ChaosScenario',
    'ChaosScenarioType',
    'ResilienceTester'
]
