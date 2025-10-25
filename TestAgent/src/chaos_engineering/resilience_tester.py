"""
Resilience Tester for TestAgentX

Tests system resilience using chaos engineering principles.
Measures recovery time, fault tolerance, and graceful degradation.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import statistics

from .chaos_orchestrator import ChaosOrchestrator, ChaosExperiment
from .chaos_scenarios import CHAOS_SCENARIOS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ResilienceMetrics:
    """Metrics for system resilience"""
    mean_time_to_recovery: float  # Seconds
    fault_tolerance_score: float  # 0-100
    graceful_degradation_score: float  # 0-100
    availability: float  # Percentage
    error_rate: float  # Percentage
    recovery_success_rate: float  # Percentage
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_time_to_recovery': self.mean_time_to_recovery,
            'fault_tolerance_score': self.fault_tolerance_score,
            'graceful_degradation_score': self.graceful_degradation_score,
            'availability': self.availability,
            'error_rate': self.error_rate,
            'recovery_success_rate': self.recovery_success_rate
        }


class ResilienceTester:
    """
    Tests and measures system resilience.
    
    Implements resilience testing patterns:
    - Fault injection
    - Recovery time measurement
    - Graceful degradation verification
    - Availability monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize resilience tester.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.orchestrator = ChaosOrchestrator(config)
        self.test_results = []
        
        logger.info("ResilienceTester initialized")
    
    def test_resilience(self,
                       system_under_test: Callable[[], Any],
                       health_check: Callable[[], bool],
                       scenarios: Optional[List[str]] = None) -> ResilienceMetrics:
        """
        Test system resilience comprehensively.
        
        Args:
            system_under_test: Function representing the system
            health_check: Function to check system health
            scenarios: List of scenario names to test (None = all)
            
        Returns:
            ResilienceMetrics with test results
        """
        logger.info("Starting comprehensive resilience testing")
        
        if scenarios is None:
            scenarios = list(CHAOS_SCENARIOS.keys())
        
        recovery_times = []
        successful_recoveries = 0
        total_tests = 0
        total_downtime = 0
        errors_encountered = 0
        
        for scenario_name in scenarios:
            scenario = CHAOS_SCENARIOS.get(scenario_name)
            if not scenario:
                continue
            
            logger.info(f"Testing resilience against: {scenario.name}")
            total_tests += 1
            
            # Measure baseline
            baseline_healthy = health_check()
            if not baseline_healthy:
                logger.warning("System not healthy at baseline")
                continue
            
            # Create experiment
            experiment = ChaosExperiment(
                name=f"Resilience Test: {scenario.name}",
                description=f"Test recovery from {scenario.name}",
                scenarios=[scenario],
                target_system="system_under_test",
                duration=scenario.duration,
                steady_state_hypothesis="System recovers within 30 seconds",
                success_criteria={'max_recovery_time': 30}
            )
            
            # Run experiment
            start_time = time.time()
            results = self.orchestrator.run_experiment(experiment, health_check)
            
            # Measure recovery time
            recovery_start = time.time()
            max_recovery_wait = 60  # seconds
            recovered = False
            
            while time.time() - recovery_start < max_recovery_wait:
                if health_check():
                    recovery_time = time.time() - recovery_start
                    recovery_times.append(recovery_time)
                    successful_recoveries += 1
                    recovered = True
                    logger.info(f"System recovered in {recovery_time:.2f}s")
                    break
                time.sleep(1)
            
            if not recovered:
                logger.warning(f"System did not recover within {max_recovery_wait}s")
                total_downtime += max_recovery_wait
                errors_encountered += 1
            else:
                total_downtime += recovery_times[-1]
        
        # Calculate metrics
        mttr = statistics.mean(recovery_times) if recovery_times else float('inf')
        recovery_rate = (successful_recoveries / total_tests * 100) if total_tests > 0 else 0
        
        # Fault tolerance score (based on recovery success)
        fault_tolerance = recovery_rate
        
        # Graceful degradation score (based on recovery time)
        if recovery_times:
            avg_recovery = statistics.mean(recovery_times)
            graceful_degradation = max(0, 100 - (avg_recovery / 30 * 100))  # 30s is target
        else:
            graceful_degradation = 0
        
        # Availability (uptime percentage)
        total_time = sum(s.duration for s in [CHAOS_SCENARIOS[name] for name in scenarios if name in CHAOS_SCENARIOS])
        availability = ((total_time - total_downtime) / total_time * 100) if total_time > 0 else 0
        
        # Error rate
        error_rate = (errors_encountered / total_tests * 100) if total_tests > 0 else 0
        
        metrics = ResilienceMetrics(
            mean_time_to_recovery=mttr,
            fault_tolerance_score=fault_tolerance,
            graceful_degradation_score=graceful_degradation,
            availability=availability,
            error_rate=error_rate,
            recovery_success_rate=recovery_rate
        )
        
        self.test_results.append(metrics)
        
        logger.info("Resilience testing complete")
        logger.info(f"MTTR: {mttr:.2f}s")
        logger.info(f"Fault Tolerance: {fault_tolerance:.2f}%")
        logger.info(f"Availability: {availability:.2f}%")
        
        return metrics
    
    def test_recovery_time(self,
                          system_under_test: Callable[[], Any],
                          health_check: Callable[[], bool],
                          scenario_name: str) -> float:
        """
        Test recovery time for a specific scenario.
        
        Args:
            system_under_test: Function representing the system
            health_check: Function to check system health
            scenario_name: Name of chaos scenario
            
        Returns:
            Recovery time in seconds
        """
        scenario = CHAOS_SCENARIOS.get(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        logger.info(f"Testing recovery time for: {scenario.name}")
        
        # Inject fault
        experiment = ChaosExperiment(
            name=f"Recovery Time Test: {scenario.name}",
            description="Measure recovery time",
            scenarios=[scenario],
            target_system="system_under_test",
            duration=scenario.duration,
            steady_state_hypothesis="System recovers quickly",
            success_criteria={'max_recovery_time': 30}
        )
        
        self.orchestrator.run_experiment(experiment, health_check)
        
        # Measure recovery
        recovery_start = time.time()
        max_wait = 60
        
        while time.time() - recovery_start < max_wait:
            if health_check():
                recovery_time = time.time() - recovery_start
                logger.info(f"Recovery time: {recovery_time:.2f}s")
                return recovery_time
            time.sleep(0.5)
        
        logger.warning(f"System did not recover within {max_wait}s")
        return float('inf')
    
    def test_fault_tolerance(self,
                            system_under_test: Callable[[], Any],
                            health_check: Callable[[], bool],
                            num_iterations: int = 10) -> float:
        """
        Test fault tolerance by running multiple chaos scenarios.
        
        Args:
            system_under_test: Function representing the system
            health_check: Function to check system health
            num_iterations: Number of test iterations
            
        Returns:
            Fault tolerance score (0-100)
        """
        logger.info(f"Testing fault tolerance with {num_iterations} iterations")
        
        successful = 0
        
        for i in range(num_iterations):
            # Select random scenario
            import random
            scenario_name = random.choice(list(CHAOS_SCENARIOS.keys()))
            scenario = CHAOS_SCENARIOS[scenario_name]
            
            # Run test
            experiment = ChaosExperiment(
                name=f"Fault Tolerance Test {i+1}",
                description="Test system fault tolerance",
                scenarios=[scenario],
                target_system="system_under_test",
                duration=scenario.duration,
                steady_state_hypothesis="System handles faults",
                success_criteria={}
            )
            
            results = self.orchestrator.run_experiment(experiment, health_check)
            
            # Check if system recovered
            time.sleep(5)  # Wait for recovery
            if health_check():
                successful += 1
        
        tolerance_score = (successful / num_iterations) * 100
        logger.info(f"Fault tolerance score: {tolerance_score:.2f}%")
        
        return tolerance_score
    
    def generate_resilience_report(self) -> str:
        """Generate a resilience testing report"""
        if not self.test_results:
            return "No resilience tests run yet."
        
        report = []
        report.append("="*60)
        report.append("RESILIENCE TESTING REPORT")
        report.append("="*60)
        report.append(f"Total Tests: {len(self.test_results)}")
        report.append("")
        
        # Calculate averages
        avg_mttr = statistics.mean(m.mean_time_to_recovery for m in self.test_results 
                                   if m.mean_time_to_recovery != float('inf'))
        avg_fault_tolerance = statistics.mean(m.fault_tolerance_score for m in self.test_results)
        avg_availability = statistics.mean(m.availability for m in self.test_results)
        
        report.append(f"Average MTTR:            {avg_mttr:.2f}s")
        report.append(f"Average Fault Tolerance: {avg_fault_tolerance:.2f}%")
        report.append(f"Average Availability:    {avg_availability:.2f}%")
        report.append("")
        
        report.append("Recommendations:")
        if avg_mttr > 30:
            report.append("  - Improve recovery mechanisms (MTTR > 30s)")
        if avg_fault_tolerance < 80:
            report.append("  - Enhance fault tolerance (score < 80%)")
        if avg_availability < 99:
            report.append("  - Increase system availability (< 99%)")
        
        if avg_mttr <= 30 and avg_fault_tolerance >= 80 and avg_availability >= 99:
            report.append("  - System shows good resilience characteristics")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    tester = ResilienceTester()
    
    # Define simple system and health check
    def my_system():
        return "OK"
    
    def health_check():
        return True
    
    # Test resilience
    metrics = tester.test_resilience(
        system_under_test=my_system,
        health_check=health_check,
        scenarios=['high_latency', 'cpu_spike']
    )
    
    print(f"Resilience Metrics:")
    print(f"  MTTR: {metrics.mean_time_to_recovery:.2f}s")
    print(f"  Fault Tolerance: {metrics.fault_tolerance_score:.2f}%")
    print(f"  Availability: {metrics.availability:.2f}%")
