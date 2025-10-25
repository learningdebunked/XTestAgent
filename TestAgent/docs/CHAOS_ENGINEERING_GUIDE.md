# Chaos Engineering for TestAgentX

## Overview

This module implements chaos engineering techniques to improve software dependability as described in **Section 3.6** of the TestAgentX paper.

Chaos engineering is the discipline of experimenting on a system to build confidence in its capability to withstand turbulent conditions in production.

## Key Components

### 1. Chaos Scenarios (`chaos_scenarios.py`)

Predefined chaos scenarios covering:

**Network Chaos**:
- High latency (500-2000ms)
- Network partitions
- Packet loss (10-30%)

**Resource Chaos**:
- CPU stress (80-100%)
- Memory leaks
- Disk exhaustion

**Application Chaos**:
- Random exceptions
- Null injections
- Slow responses

**State Chaos**:
- State corruption
- Race conditions
- Deadlocks

**Time Chaos**:
- Time skew
- Clock drift

### 2. Fault Injector (`fault_injector.py`)

Injects faults into the system:
- Network manipulation
- Resource exhaustion
- Exception injection
- State corruption
- Time manipulation

### 3. Chaos Orchestrator (`chaos_orchestrator.py`)

Orchestrates chaos experiments:
- Defines steady state
- Runs chaos scenarios
- Verifies recovery
- Generates reports

### 4. Resilience Tester (`resilience_tester.py`)

Measures system resilience:
- Mean Time To Recovery (MTTR)
- Fault tolerance score
- Availability percentage
- Graceful degradation

## Quick Start

### Basic Chaos Experiment

```python
from chaos_engineering import ChaosOrchestrator, ChaosExperiment
from chaos_engineering.chaos_scenarios import CHAOS_SCENARIOS

# Initialize orchestrator
orchestrator = ChaosOrchestrator()

# Define health check
def health_check():
    # Check if system is healthy
    return system.is_responsive()

# Create experiment
experiment = ChaosExperiment(
    name="Network Resilience Test",
    description="Test system resilience to network issues",
    scenarios=[
        CHAOS_SCENARIOS['high_latency'],
        CHAOS_SCENARIOS['packet_loss']
    ],
    target_system="my_application",
    duration=60,
    steady_state_hypothesis="System maintains < 1s response time",
    success_criteria={'response_time_ms': 1000}
)

# Run experiment
results = orchestrator.run_experiment(experiment, health_check)
print(f"Experiment successful: {results['experiment_successful']}")
```

### Resilience Testing

```python
from chaos_engineering import ResilienceTester

# Initialize tester
tester = ResilienceTester()

# Test resilience
metrics = tester.test_resilience(
    system_under_test=my_system,
    health_check=health_check,
    scenarios=['high_latency', 'cpu_spike', 'memory_leak']
)

print(f"MTTR: {metrics.mean_time_to_recovery:.2f}s")
print(f"Fault Tolerance: {metrics.fault_tolerance_score:.2f}%")
print(f"Availability: {metrics.availability:.2f}%")
```

### Continuous Chaos

```python
# Run continuous chaos for 1 hour
orchestrator.run_continuous_chaos(
    duration=3600,  # 1 hour
    interval=60,    # Every minute
    severity_range=(1, 3)  # Low to medium severity
)
```

### Chaos Gameday

```python
# Schedule a chaos gameday
results = orchestrator.run_gameday(
    scenarios=[
        'high_latency',
        'cpu_spike',
        'random_exceptions',
        'packet_loss'
    ],
    system_health_check=health_check
)
```

## Available Chaos Scenarios

| Scenario | Type | Severity | Description |
|----------|------|----------|-------------|
| `high_latency` | Network | 3 | 500-2000ms latency |
| `network_partition` | Network | 5 | Network split |
| `packet_loss` | Network | 4 | 10-30% packet loss |
| `cpu_spike` | Resource | 4 | 80-100% CPU usage |
| `memory_leak` | Resource | 4 | Gradual memory consumption |
| `disk_full` | Resource | 5 | Fill disk to 95% |
| `random_exceptions` | Application | 3 | Random exceptions |
| `null_injection` | Application | 3 | Return nulls randomly |
| `slow_method` | Application | 2 | Add delays to methods |
| `state_corruption` | State | 5 | Corrupt application state |
| `race_condition` | State | 4 | Trigger race conditions |
| `time_skew` | Time | 3 | Shift system time |
| `clock_drift` | Time | 2 | Clock runs faster/slower |

## Custom Scenarios

Create custom chaos scenarios:

```python
from chaos_engineering.chaos_scenarios import ChaosScenario, ChaosScenarioType

custom_scenario = ChaosScenario(
    scenario_type=ChaosScenarioType.EXCEPTION_INJECTION,
    name="Database Timeout",
    description="Simulate database connection timeouts",
    severity=4,
    duration=30.0,
    probability=0.5,
    parameters={
        'exception_types': ['TimeoutException'],
        'injection_rate': 0.2
    },
    target_components=['database_layer']
)
```

## Measuring Resilience

### Key Metrics

1. **Mean Time To Recovery (MTTR)**
   - Average time to recover from failures
   - Target: < 30 seconds

2. **Fault Tolerance Score**
   - Percentage of faults handled gracefully
   - Target: > 80%

3. **Availability**
   - Uptime percentage during chaos
   - Target: > 99%

4. **Graceful Degradation**
   - How well system degrades under stress
   - Target: > 80%

### Example Measurement

```python
# Test specific scenario
recovery_time = tester.test_recovery_time(
    system_under_test=my_system,
    health_check=health_check,
    scenario_name='cpu_spike'
)

print(f"Recovery time: {recovery_time:.2f}s")

# Test fault tolerance
tolerance = tester.test_fault_tolerance(
    system_under_test=my_system,
    health_check=health_check,
    num_iterations=20
)

print(f"Fault tolerance: {tolerance:.2f}%")
```

## Integration with TestAgentX

### In Test Generation

```python
from layer2_test_generation.llm_test_agent import LLMTestGenerationAgent
from chaos_engineering import ChaosOrchestrator

# Generate tests
agent = LLMTestGenerationAgent()
tests = agent.generate_tests(...)

# Test resilience of generated tests
orchestrator = ChaosOrchestrator()
experiment = ChaosExperiment(
    name="Test Suite Resilience",
    description="Verify tests handle chaos",
    scenarios=[CHAOS_SCENARIOS['random_exceptions']],
    target_system="test_suite",
    duration=30,
    steady_state_hypothesis="Tests pass under chaos",
    success_criteria={'pass_rate': 0.95}
)

results = orchestrator.run_experiment(experiment)
```

### In Patch Verification

```python
from layer4_patch_regression.patch_verification_agent import PatchVerificationAgent
from chaos_engineering import ResilienceTester

# Verify patch
patch_agent = PatchVerificationAgent()
result = patch_agent.verify_patch(...)

# Test patch resilience
tester = ResilienceTester()
metrics = tester.test_resilience(
    system_under_test=patched_system,
    health_check=lambda: patched_system.is_healthy(),
    scenarios=['cpu_spike', 'memory_leak']
)

print(f"Patch resilience score: {metrics.fault_tolerance_score:.2f}%")
```

## Best Practices

### 1. Start Small
Begin with low-severity scenarios and gradually increase:
```python
# Week 1: Low severity
scenarios = get_scenarios_by_severity(1, 2)

# Week 2: Medium severity
scenarios = get_scenarios_by_severity(2, 3)

# Week 3: High severity
scenarios = get_scenarios_by_severity(3, 5)
```

### 2. Define Clear Hypotheses
Always state what you expect to happen:
```python
experiment = ChaosExperiment(
    ...
    steady_state_hypothesis="API response time < 500ms under 10% packet loss",
    success_criteria={'max_response_ms': 500}
)
```

### 3. Monitor Everything
Implement comprehensive health checks:
```python
def health_check():
    checks = {
        'api_responsive': check_api(),
        'database_connected': check_database(),
        'queue_processing': check_queue(),
        'memory_ok': check_memory()
    }
    return all(checks.values())
```

### 4. Learn and Improve
Use results to improve system:
```python
results = orchestrator.run_experiment(experiment, health_check)

for recommendation in results['recommendations']:
    print(f"TODO: {recommendation}")
```

### 5. Automate
Run chaos tests regularly:
```bash
# Cron job for daily chaos
0 2 * * * python -m chaos_engineering.run_daily_chaos
```

## Troubleshooting

### Scenario Not Triggering

Check probability setting:
```python
scenario.probability = 1.0  # Always trigger
```

### System Not Recovering

Increase recovery wait time:
```python
max_recovery_wait = 120  # 2 minutes
```

### High Resource Usage

Reduce scenario severity:
```python
scenario.parameters['cpu_percentage'] = 50  # Lower from 90%
```

## Example: Full Resilience Test

```python
#!/usr/bin/env python3
"""
Complete resilience testing example
"""

from chaos_engineering import (
    ChaosOrchestrator,
    ResilienceTester,
    ChaosExperiment
)
from chaos_engineering.chaos_scenarios import CHAOS_SCENARIOS

def main():
    # Initialize
    orchestrator = ChaosOrchestrator()
    tester = ResilienceTester()
    
    # Define system
    def my_system():
        # Your system logic
        return "OK"
    
    def health_check():
        # Your health check logic
        try:
            response = my_system()
            return response == "OK"
        except:
            return False
    
    # Test 1: Network resilience
    print("Testing network resilience...")
    network_experiment = ChaosExperiment(
        name="Network Resilience",
        description="Test network fault handling",
        scenarios=[
            CHAOS_SCENARIOS['high_latency'],
            CHAOS_SCENARIOS['packet_loss']
        ],
        target_system="application",
        duration=60,
        steady_state_hypothesis="System handles network issues",
        success_criteria={'max_downtime': 10}
    )
    
    network_results = orchestrator.run_experiment(
        network_experiment,
        health_check
    )
    
    # Test 2: Resource resilience
    print("Testing resource resilience...")
    resource_metrics = tester.test_resilience(
        system_under_test=my_system,
        health_check=health_check,
        scenarios=['cpu_spike', 'memory_leak']
    )
    
    # Generate report
    print("\n" + "="*60)
    print("RESILIENCE TEST RESULTS")
    print("="*60)
    print(f"Network Test: {'PASS' if network_results['experiment_successful'] else 'FAIL'}")
    print(f"MTTR: {resource_metrics.mean_time_to_recovery:.2f}s")
    print(f"Fault Tolerance: {resource_metrics.fault_tolerance_score:.2f}%")
    print(f"Availability: {resource_metrics.availability:.2f}%")
    
    # Recommendations
    print("\nRecommendations:")
    for rec in network_results.get('recommendations', []):
        print(f"  - {rec}")

if __name__ == "__main__":
    main()
```

## References

- Principles of Chaos Engineering: https://principlesofchaos.org/
- Netflix Chaos Monkey: https://netflix.github.io/chaosmonkey/
- TestAgentX Paper: Section 3.6 (Chaos Engineering)

## Support

For questions about chaos engineering:
1. Review this guide
2. Check example scripts in `examples/chaos_engineering/`
3. See test results in `chaos_results/`
4. Open an issue on GitHub
