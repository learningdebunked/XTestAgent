"""
Chaos Orchestrator for TestAgentX

Orchestrates chaos engineering experiments to test system resilience.
Implements Section 3.6 of the paper: Chaos Engineering for Dependability.
"""

import logging
import time
import random
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from .chaos_scenarios import ChaosScenario, ChaosScenarioType, CHAOS_SCENARIOS
from .fault_injector import FaultInjector, InjectionResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChaosExperiment:
    """Represents a chaos engineering experiment"""
    name: str
    description: str
    scenarios: List[ChaosScenario]
    target_system: str
    duration: float  # Total experiment duration in seconds
    steady_state_hypothesis: str
    success_criteria: Dict[str, Any]
    results: List[InjectionResult] = field(default_factory=list)
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'description': self.description,
            'scenarios': [s.to_dict() for s in self.scenarios],
            'target_system': self.target_system,
            'duration': self.duration,
            'steady_state_hypothesis': self.steady_state_hypothesis,
            'success_criteria': self.success_criteria,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'num_results': len(self.results)
        }


class ChaosOrchestrator:
    """
    Orchestrates chaos engineering experiments.
    
    Implements the chaos engineering methodology:
    1. Define steady state
    2. Hypothesize steady state continues
    3. Introduce chaos
    4. Verify steady state
    5. Learn and improve
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize chaos orchestrator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.fault_injector = FaultInjector(config)
        self.experiments = []
        self.results_dir = Path(self.config.get('results_dir', 'chaos_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("ChaosOrchestrator initialized")
    
    def run_experiment(self, experiment: ChaosExperiment,
                      system_health_check: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """
        Run a chaos engineering experiment.
        
        Args:
            experiment: Chaos experiment to run
            system_health_check: Function to check system health
            
        Returns:
            Dictionary with experiment results
        """
        logger.info(f"Starting chaos experiment: {experiment.name}")
        logger.info(f"Hypothesis: {experiment.steady_state_hypothesis}")
        
        experiment.start_time = datetime.now().isoformat()
        
        # 1. Verify steady state before chaos
        if system_health_check:
            if not system_health_check():
                logger.error("System not in steady state before experiment")
                return {
                    'success': False,
                    'error': 'System not in steady state'
                }
        
        # 2. Run chaos scenarios
        results = []
        for scenario in experiment.scenarios:
            if scenario.should_trigger():
                logger.info(f"Triggering scenario: {scenario.name}")
                
                # Inject fault
                result = self.fault_injector.inject_fault(scenario)
                results.append(result)
                
                # Check system health during chaos
                if system_health_check:
                    health_ok = system_health_check()
                    result.impact_metrics['health_check_passed'] = health_ok
                    
                    if not health_ok:
                        logger.warning(f"Health check failed during {scenario.name}")
                
                # Wait between scenarios
                time.sleep(2)
            else:
                logger.info(f"Skipping scenario (probability): {scenario.name}")
        
        experiment.results = results
        experiment.end_time = datetime.now().isoformat()
        
        # 3. Verify steady state after chaos
        steady_state_recovered = True
        if system_health_check:
            # Wait for recovery
            time.sleep(5)
            steady_state_recovered = system_health_check()
        
        # 4. Analyze results
        analysis = self._analyze_experiment(experiment, steady_state_recovered)
        
        # 5. Save results
        self._save_experiment_results(experiment, analysis)
        
        self.experiments.append(experiment)
        
        logger.info(f"Chaos experiment complete: {experiment.name}")
        
        return analysis
    
    def run_continuous_chaos(self, duration: float = 3600,
                           interval: float = 60,
                           severity_range: tuple = (1, 3)) -> None:
        """
        Run continuous chaos testing.
        
        Args:
            duration: Total duration in seconds
            interval: Time between chaos injections
            severity_range: Min and max severity levels to use
        """
        logger.info(f"Starting continuous chaos for {duration}s")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            logger.info(f"Continuous chaos iteration {iteration}")
            
            # Select random scenario within severity range
            scenarios = [s for s in CHAOS_SCENARIOS.values()
                        if severity_range[0] <= s.severity <= severity_range[1]]
            
            if scenarios:
                scenario = random.choice(scenarios)
                
                # Create mini experiment
                experiment = ChaosExperiment(
                    name=f"Continuous Chaos {iteration}",
                    description=f"Iteration {iteration} of continuous chaos",
                    scenarios=[scenario],
                    target_system="application",
                    duration=scenario.duration,
                    steady_state_hypothesis="System remains responsive",
                    success_criteria={'response_time_ms': 1000}
                )
                
                self.run_experiment(experiment)
            
            # Wait before next iteration
            time.sleep(interval)
        
        logger.info("Continuous chaos testing complete")
    
    def run_gameday(self, scenarios: List[str],
                   system_health_check: Callable[[], bool]) -> Dict[str, Any]:
        """
        Run a chaos gameday with predefined scenarios.
        
        Args:
            scenarios: List of scenario names to run
            system_health_check: Function to check system health
            
        Returns:
            Gameday results
        """
        logger.info(f"Starting chaos gameday with {len(scenarios)} scenarios")
        
        # Build experiment
        chaos_scenarios = []
        for scenario_name in scenarios:
            scenario = CHAOS_SCENARIOS.get(scenario_name)
            if scenario:
                chaos_scenarios.append(scenario)
            else:
                logger.warning(f"Unknown scenario: {scenario_name}")
        
        experiment = ChaosExperiment(
            name="Chaos Gameday",
            description="Scheduled chaos engineering gameday",
            scenarios=chaos_scenarios,
            target_system="application",
            duration=sum(s.duration for s in chaos_scenarios),
            steady_state_hypothesis="System handles all chaos scenarios gracefully",
            success_criteria={
                'max_downtime_seconds': 30,
                'max_error_rate': 0.05
            }
        )
        
        return self.run_experiment(experiment, system_health_check)
    
    def _analyze_experiment(self, experiment: ChaosExperiment,
                          steady_state_recovered: bool) -> Dict[str, Any]:
        """Analyze experiment results"""
        total_scenarios = len(experiment.scenarios)
        successful_injections = sum(1 for r in experiment.results if r.success)
        failed_injections = total_scenarios - successful_injections
        
        total_errors = sum(len(r.errors_caught) for r in experiment.results)
        
        # Calculate average impact
        avg_cpu = 0
        avg_memory = 0
        if experiment.results:
            avg_cpu = sum(r.impact_metrics.get('cpu_usage', 0) 
                         for r in experiment.results) / len(experiment.results)
            avg_memory = sum(r.impact_metrics.get('memory_usage', 0) 
                           for r in experiment.results) / len(experiment.results)
        
        # Determine if experiment met success criteria
        experiment_successful = (
            steady_state_recovered and
            successful_injections >= total_scenarios * 0.8  # 80% success rate
        )
        
        analysis = {
            'experiment_name': experiment.name,
            'experiment_successful': experiment_successful,
            'steady_state_recovered': steady_state_recovered,
            'total_scenarios': total_scenarios,
            'successful_injections': successful_injections,
            'failed_injections': failed_injections,
            'total_errors_caught': total_errors,
            'average_cpu_impact': avg_cpu,
            'average_memory_impact': avg_memory,
            'duration': experiment.duration,
            'recommendations': self._generate_recommendations(experiment)
        }
        
        return analysis
    
    def _generate_recommendations(self, experiment: ChaosExperiment) -> List[str]:
        """Generate recommendations based on experiment results"""
        recommendations = []
        
        # Check for high failure rate
        failure_rate = sum(1 for r in experiment.results if not r.success) / len(experiment.results) if experiment.results else 0
        if failure_rate > 0.2:
            recommendations.append(
                f"High failure rate ({failure_rate*100:.1f}%). "
                "Consider improving error handling and resilience mechanisms."
            )
        
        # Check for high resource usage
        for result in experiment.results:
            cpu = result.impact_metrics.get('cpu_usage', 0)
            memory = result.impact_metrics.get('memory_usage', 0)
            
            if cpu > 90:
                recommendations.append(
                    f"High CPU usage ({cpu}%) during {result.scenario_name}. "
                    "Consider implementing rate limiting or circuit breakers."
                )
            
            if memory > 90:
                recommendations.append(
                    f"High memory usage ({memory}%) during {result.scenario_name}. "
                    "Check for memory leaks and implement proper resource cleanup."
                )
        
        # Check for unhandled errors
        total_errors = sum(len(r.errors_caught) for r in experiment.results)
        if total_errors > 0:
            recommendations.append(
                f"{total_errors} errors caught during experiment. "
                "Review error logs and add appropriate error handling."
            )
        
        if not recommendations:
            recommendations.append("System handled chaos scenarios well. Continue monitoring.")
        
        return recommendations
    
    def _save_experiment_results(self, experiment: ChaosExperiment,
                                analysis: Dict[str, Any]) -> None:
        """Save experiment results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chaos_experiment_{timestamp}.json"
        filepath = self.results_dir / filename
        
        data = {
            'experiment': experiment.to_dict(),
            'analysis': analysis,
            'results': [
                {
                    'scenario': r.scenario_name,
                    'success': r.success,
                    'duration': r.duration,
                    'impact_metrics': r.impact_metrics,
                    'errors': r.errors_caught
                }
                for r in experiment.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Experiment results saved to {filepath}")
    
    def get_experiment_history(self) -> List[ChaosExperiment]:
        """Get history of all experiments"""
        return self.experiments
    
    def generate_report(self) -> str:
        """Generate a summary report of all experiments"""
        if not self.experiments:
            return "No experiments run yet."
        
        report = []
        report.append("="*60)
        report.append("CHAOS ENGINEERING SUMMARY REPORT")
        report.append("="*60)
        report.append(f"Total Experiments: {len(self.experiments)}")
        report.append("")
        
        for exp in self.experiments:
            report.append(f"Experiment: {exp.name}")
            report.append(f"  Scenarios: {len(exp.scenarios)}")
            report.append(f"  Duration: {exp.duration}s")
            report.append(f"  Results: {len(exp.results)} injections")
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    orchestrator = ChaosOrchestrator()
    
    # Define a simple health check
    def health_check():
        return True  # Simplified
    
    # Create an experiment
    experiment = ChaosExperiment(
        name="Network Resilience Test",
        description="Test system resilience to network issues",
        scenarios=[
            CHAOS_SCENARIOS['high_latency'],
            CHAOS_SCENARIOS['packet_loss']
        ],
        target_system="test_application",
        duration=60,
        steady_state_hypothesis="System maintains < 1s response time",
        success_criteria={'response_time_ms': 1000}
    )
    
    # Run experiment
    results = orchestrator.run_experiment(experiment, health_check)
    print(json.dumps(results, indent=2))
