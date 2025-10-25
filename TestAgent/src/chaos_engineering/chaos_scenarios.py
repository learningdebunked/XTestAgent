"""
Chaos Scenarios for TestAgentX

Defines various chaos engineering scenarios to test system resilience.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import random


class ChaosScenarioType(Enum):
    """Types of chaos scenarios"""
    # Network chaos
    NETWORK_LATENCY = "network_latency"
    NETWORK_PARTITION = "network_partition"
    PACKET_LOSS = "packet_loss"
    
    # Resource chaos
    CPU_STRESS = "cpu_stress"
    MEMORY_STRESS = "memory_stress"
    DISK_STRESS = "disk_stress"
    
    # Application chaos
    EXCEPTION_INJECTION = "exception_injection"
    NULL_INJECTION = "null_injection"
    SLOW_RESPONSE = "slow_response"
    
    # State chaos
    STATE_CORRUPTION = "state_corruption"
    RACE_CONDITION = "race_condition"
    DEADLOCK = "deadlock"
    
    # Time chaos
    TIME_SKEW = "time_skew"
    CLOCK_DRIFT = "clock_drift"


@dataclass
class ChaosScenario:
    """Represents a chaos engineering scenario"""
    scenario_type: ChaosScenarioType
    name: str
    description: str
    severity: int  # 1-5, where 5 is most severe
    duration: float  # Seconds
    probability: float  # 0.0-1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_components: List[str] = field(default_factory=list)
    
    def should_trigger(self) -> bool:
        """Determine if scenario should trigger based on probability"""
        return random.random() < self.probability
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scenario_type': self.scenario_type.value,
            'name': self.name,
            'description': self.description,
            'severity': self.severity,
            'duration': self.duration,
            'probability': self.probability,
            'parameters': self.parameters,
            'target_components': self.target_components
        }


# Predefined chaos scenarios
CHAOS_SCENARIOS = {
    # Network Chaos
    'high_latency': ChaosScenario(
        scenario_type=ChaosScenarioType.NETWORK_LATENCY,
        name="High Network Latency",
        description="Inject 500ms-2000ms latency into network calls",
        severity=3,
        duration=30.0,
        probability=0.3,
        parameters={'min_latency_ms': 500, 'max_latency_ms': 2000}
    ),
    
    'network_partition': ChaosScenario(
        scenario_type=ChaosScenarioType.NETWORK_PARTITION,
        name="Network Partition",
        description="Simulate network split between components",
        severity=5,
        duration=15.0,
        probability=0.1,
        parameters={'partition_duration': 15}
    ),
    
    'packet_loss': ChaosScenario(
        scenario_type=ChaosScenarioType.PACKET_LOSS,
        name="Packet Loss",
        description="Drop 10-30% of network packets",
        severity=4,
        duration=20.0,
        probability=0.2,
        parameters={'loss_percentage': 20}
    ),
    
    # Resource Chaos
    'cpu_spike': ChaosScenario(
        scenario_type=ChaosScenarioType.CPU_STRESS,
        name="CPU Spike",
        description="Consume 80-100% CPU for short period",
        severity=4,
        duration=10.0,
        probability=0.25,
        parameters={'cpu_percentage': 90, 'cores': 2}
    ),
    
    'memory_leak': ChaosScenario(
        scenario_type=ChaosScenarioType.MEMORY_STRESS,
        name="Memory Leak Simulation",
        description="Gradually consume memory to simulate leak",
        severity=4,
        duration=30.0,
        probability=0.15,
        parameters={'memory_mb': 512, 'rate_mb_per_sec': 10}
    ),
    
    'disk_full': ChaosScenario(
        scenario_type=ChaosScenarioType.DISK_STRESS,
        name="Disk Full",
        description="Fill disk to 95% capacity",
        severity=5,
        duration=20.0,
        probability=0.1,
        parameters={'fill_percentage': 95}
    ),
    
    # Application Chaos
    'random_exceptions': ChaosScenario(
        scenario_type=ChaosScenarioType.EXCEPTION_INJECTION,
        name="Random Exceptions",
        description="Inject random exceptions into method calls",
        severity=3,
        duration=25.0,
        probability=0.3,
        parameters={
            'exception_types': ['NullPointerException', 'IllegalArgumentException', 'RuntimeException'],
            'injection_rate': 0.1
        }
    ),
    
    'null_injection': ChaosScenario(
        scenario_type=ChaosScenarioType.NULL_INJECTION,
        name="Null Injection",
        description="Return null from methods randomly",
        severity=3,
        duration=20.0,
        probability=0.25,
        parameters={'injection_rate': 0.15}
    ),
    
    'slow_method': ChaosScenario(
        scenario_type=ChaosScenarioType.SLOW_RESPONSE,
        name="Slow Method Execution",
        description="Add random delays to method execution",
        severity=2,
        duration=30.0,
        probability=0.4,
        parameters={'min_delay_ms': 100, 'max_delay_ms': 1000}
    ),
    
    # State Chaos
    'state_corruption': ChaosScenario(
        scenario_type=ChaosScenarioType.STATE_CORRUPTION,
        name="State Corruption",
        description="Corrupt application state randomly",
        severity=5,
        duration=15.0,
        probability=0.1,
        parameters={'corruption_rate': 0.05}
    ),
    
    'race_condition': ChaosScenario(
        scenario_type=ChaosScenarioType.RACE_CONDITION,
        name="Race Condition Trigger",
        description="Introduce timing issues to trigger race conditions",
        severity=4,
        duration=20.0,
        probability=0.15,
        parameters={'thread_count': 10, 'contention_points': 5}
    ),
    
    # Time Chaos
    'time_skew': ChaosScenario(
        scenario_type=ChaosScenarioType.TIME_SKEW,
        name="Time Skew",
        description="Shift system time forward/backward",
        severity=3,
        duration=25.0,
        probability=0.2,
        parameters={'skew_seconds': 3600}  # 1 hour
    ),
    
    'clock_drift': ChaosScenario(
        scenario_type=ChaosScenarioType.CLOCK_DRIFT,
        name="Clock Drift",
        description="Simulate clock running faster/slower",
        severity=2,
        duration=30.0,
        probability=0.25,
        parameters={'drift_factor': 1.5}  # 1.5x speed
    )
}


def get_scenario(scenario_name: str) -> Optional[ChaosScenario]:
    """Get a predefined chaos scenario by name"""
    return CHAOS_SCENARIOS.get(scenario_name)


def get_scenarios_by_type(scenario_type: ChaosScenarioType) -> List[ChaosScenario]:
    """Get all scenarios of a specific type"""
    return [s for s in CHAOS_SCENARIOS.values() if s.scenario_type == scenario_type]


def get_scenarios_by_severity(min_severity: int, max_severity: int = 5) -> List[ChaosScenario]:
    """Get scenarios within a severity range"""
    return [s for s in CHAOS_SCENARIOS.values() 
            if min_severity <= s.severity <= max_severity]
