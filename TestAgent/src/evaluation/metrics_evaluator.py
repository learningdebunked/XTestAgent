"""
Metrics Evaluator for TestAgentX

Implements evaluation metrics to validate paper claims:
- Test coverage measurement
- Mutation score calculation
- Test generation time benchmarking
- Patch verification accuracy
- False positive rate tracking
- Developer acceptance metrics
"""

import time
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import xml.etree.ElementTree as ET
from datetime import datetime
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Test coverage metrics"""
    line_coverage: float  # Percentage
    branch_coverage: float  # Percentage
    method_coverage: float  # Percentage
    class_coverage: float  # Percentage
    total_lines: int
    covered_lines: int
    total_branches: int
    covered_branches: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MutationMetrics:
    """Mutation testing metrics"""
    mutation_score: float  # Percentage
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    timeout_mutants: int
    equivalent_mutants: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Test generation performance metrics"""
    total_time: float  # Seconds
    tests_generated: int
    time_per_test: float  # Seconds
    baseline_time: Optional[float] = None  # Baseline comparison
    time_reduction: Optional[float] = None  # Percentage
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccuracyMetrics:
    """Patch verification accuracy metrics"""
    accuracy: float  # Percentage
    precision: float  # Percentage
    recall: float  # Percentage
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    false_positive_rate: float  # Percentage
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeveloperAcceptanceMetrics:
    """Developer acceptance metrics"""
    acceptance_rate: float  # Percentage
    total_tests: int
    accepted_tests: int
    rejected_tests: int
    modified_tests: int
    average_quality_score: float  # 1-5 scale
    feedback_comments: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsEvaluator:
    """
    Main evaluator for measuring TestAgentX performance metrics.
    Validates claims from the paper.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize metrics evaluator.
        
        Args:
            config: Configuration dictionary with paths and settings
        """
        self.config = config or {}
        self.results_dir = Path(self.config.get('results_dir', 'evaluation_results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Tool paths
        self.jacoco_cli = self.config.get('jacoco_cli', 'lib/jacococli.jar')
        self.pitest_jar = self.config.get('pitest_jar', 'lib/pitest.jar')
        
        logger.info("MetricsEvaluator initialized")
    
    def measure_test_coverage(self, project_path: str, 
                             test_classes: List[str]) -> CoverageMetrics:
        """
        Measure test coverage using JaCoCo.
        
        Target: 89% test coverage (paper claim)
        
        Args:
            project_path: Path to the project
            test_classes: List of test class names
            
        Returns:
            CoverageMetrics with coverage data
        """
        logger.info("Measuring test coverage...")
        
        try:
            # Run tests with JaCoCo
            jacoco_exec = Path(project_path) / 'target' / 'jacoco.exec'
            
            # Execute tests with coverage
            cmd = [
                'mvn', 'clean', 'test',
                f'-Djacoco.destFile={jacoco_exec}'
            ]
            
            subprocess.run(cmd, cwd=project_path, check=True, capture_output=True)
            
            # Generate XML report
            xml_report = self.results_dir / 'coverage_report.xml'
            self._generate_jacoco_report(project_path, jacoco_exec, xml_report)
            
            # Parse coverage data
            metrics = self._parse_coverage_report(xml_report)
            
            # Save results
            self._save_metrics(metrics, 'coverage_metrics.json')
            
            logger.info(f"Coverage: {metrics.line_coverage:.2f}% lines, "
                       f"{metrics.branch_coverage:.2f}% branches")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error measuring coverage: {e}", exc_info=True)
            raise
    
    def measure_mutation_score(self, project_path: str,
                               target_classes: List[str]) -> MutationMetrics:
        """
        Measure mutation score using PIT (PITest).
        
        Target: 84% mutation score (paper claim)
        
        Args:
            project_path: Path to the project
            target_classes: List of classes to mutate
            
        Returns:
            MutationMetrics with mutation testing data
        """
        logger.info("Measuring mutation score...")
        
        try:
            # Run PITest
            cmd = [
                'mvn', 'org.pitest:pitest-maven:mutationCoverage',
                '-DtargetClasses=' + ','.join(target_classes),
                '-DoutputFormats=XML,HTML'
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=project_path, 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Parse mutation report
            mutation_report = Path(project_path) / 'target' / 'pit-reports' / 'mutations.xml'
            metrics = self._parse_mutation_report(mutation_report)
            
            # Save results
            self._save_metrics(metrics, 'mutation_metrics.json')
            
            logger.info(f"Mutation score: {metrics.mutation_score:.2f}% "
                       f"({metrics.killed_mutants}/{metrics.total_mutants} killed)")
            
            return metrics
            
        except subprocess.TimeoutExpired:
            logger.error("Mutation testing timed out")
            raise
        except Exception as e:
            logger.error(f"Error measuring mutation score: {e}", exc_info=True)
            raise
    
    def benchmark_test_generation(self, 
                                 test_generation_fn: callable,
                                 baseline_fn: Optional[callable] = None,
                                 num_iterations: int = 10) -> PerformanceMetrics:
        """
        Benchmark test generation time.
        
        Target: 55% reduction in test generation time (paper claim)
        
        Args:
            test_generation_fn: Function that generates tests
            baseline_fn: Baseline function for comparison (e.g., EvoSuite)
            num_iterations: Number of iterations for averaging
            
        Returns:
            PerformanceMetrics with timing data
        """
        logger.info("Benchmarking test generation performance...")
        
        # Measure TestAgentX time
        testagentx_times = []
        tests_generated = 0
        
        for i in range(num_iterations):
            start_time = time.time()
            result = test_generation_fn()
            elapsed = time.time() - start_time
            
            testagentx_times.append(elapsed)
            tests_generated += len(result) if isinstance(result, list) else 1
        
        avg_time = statistics.mean(testagentx_times)
        tests_per_iter = tests_generated / num_iterations
        time_per_test = avg_time / tests_per_iter if tests_per_iter > 0 else 0
        
        # Measure baseline if provided
        baseline_time = None
        time_reduction = None
        
        if baseline_fn:
            baseline_times = []
            for i in range(num_iterations):
                start_time = time.time()
                baseline_fn()
                elapsed = time.time() - start_time
                baseline_times.append(elapsed)
            
            baseline_time = statistics.mean(baseline_times)
            time_reduction = ((baseline_time - avg_time) / baseline_time) * 100
        
        metrics = PerformanceMetrics(
            total_time=avg_time,
            tests_generated=int(tests_per_iter),
            time_per_test=time_per_test,
            baseline_time=baseline_time,
            time_reduction=time_reduction
        )
        
        # Save results
        self._save_metrics(metrics, 'performance_metrics.json')
        
        logger.info(f"Average time: {avg_time:.2f}s, "
                   f"Time per test: {time_per_test:.2f}s")
        if time_reduction:
            logger.info(f"Time reduction vs baseline: {time_reduction:.2f}%")
        
        return metrics
    
    def measure_patch_verification_accuracy(self,
                                          verification_results: List[Dict[str, Any]],
                                          ground_truth: List[Dict[str, Any]]) -> AccuracyMetrics:
        """
        Measure patch verification accuracy.
        
        Target: 91% accuracy, 8% false positive rate (paper claims)
        
        Args:
            verification_results: List of verification results from agent
            ground_truth: List of ground truth labels
            
        Returns:
            AccuracyMetrics with accuracy data
        """
        logger.info("Measuring patch verification accuracy...")
        
        if len(verification_results) != len(ground_truth):
            raise ValueError("Results and ground truth must have same length")
        
        tp = tn = fp = fn = 0
        
        for result, truth in zip(verification_results, ground_truth):
            predicted = result['is_effective']
            actual = truth['is_effective']
            
            if predicted and actual:
                tp += 1
            elif not predicted and not actual:
                tn += 1
            elif predicted and not actual:
                fp += 1
            else:  # not predicted and actual
                fn += 1
        
        total = len(verification_results)
        accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0
        
        metrics = AccuracyMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            false_positive_rate=fpr
        )
        
        # Save results
        self._save_metrics(metrics, 'accuracy_metrics.json')
        
        logger.info(f"Accuracy: {accuracy:.2f}%, FPR: {fpr:.2f}%")
        logger.info(f"Precision: {precision:.2f}%, Recall: {recall:.2f}%")
        
        return metrics
    
    def measure_developer_acceptance(self,
                                    generated_tests: List[Dict[str, Any]],
                                    developer_feedback: List[Dict[str, Any]]) -> DeveloperAcceptanceMetrics:
        """
        Measure developer acceptance rate.
        
        Target: 82% developer acceptance (paper claim)
        
        Args:
            generated_tests: List of generated test cases
            developer_feedback: List of developer feedback entries
            
        Returns:
            DeveloperAcceptanceMetrics with acceptance data
        """
        logger.info("Measuring developer acceptance...")
        
        accepted = 0
        rejected = 0
        modified = 0
        quality_scores = []
        comments = []
        
        for feedback in developer_feedback:
            status = feedback.get('status', 'unknown')
            
            if status == 'accepted':
                accepted += 1
            elif status == 'rejected':
                rejected += 1
            elif status == 'modified':
                modified += 1
                accepted += 1  # Modified tests are still accepted
            
            if 'quality_score' in feedback:
                quality_scores.append(feedback['quality_score'])
            
            if 'comment' in feedback:
                comments.append(feedback['comment'])
        
        total = len(developer_feedback)
        acceptance_rate = (accepted / total) * 100 if total > 0 else 0
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        metrics = DeveloperAcceptanceMetrics(
            acceptance_rate=acceptance_rate,
            total_tests=total,
            accepted_tests=accepted,
            rejected_tests=rejected,
            modified_tests=modified,
            average_quality_score=avg_quality,
            feedback_comments=comments[:10]  # Store first 10 comments
        )
        
        # Save results
        self._save_metrics(metrics, 'acceptance_metrics.json')
        
        logger.info(f"Acceptance rate: {acceptance_rate:.2f}%")
        logger.info(f"Average quality score: {avg_quality:.2f}/5.0")
        
        return metrics
    
    def _generate_jacoco_report(self, project_path: str, 
                               exec_file: Path, output_xml: Path) -> None:
        """Generate JaCoCo XML report from exec file"""
        cmd = [
            'java', '-jar', self.jacoco_cli, 'report', str(exec_file),
            '--classfiles', f'{project_path}/target/classes',
            '--sourcefiles', f'{project_path}/src/main/java',
            '--xml', str(output_xml)
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _parse_coverage_report(self, xml_file: Path) -> CoverageMetrics:
        """Parse JaCoCo XML report"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract counter elements
        counters = {}
        for counter in root.findall('.//counter'):
            type_name = counter.get('type')
            covered = int(counter.get('covered', 0))
            missed = int(counter.get('missed', 0))
            total = covered + missed
            coverage = (covered / total * 100) if total > 0 else 0
            counters[type_name] = {
                'covered': covered,
                'total': total,
                'coverage': coverage
            }
        
        return CoverageMetrics(
            line_coverage=counters.get('LINE', {}).get('coverage', 0),
            branch_coverage=counters.get('BRANCH', {}).get('coverage', 0),
            method_coverage=counters.get('METHOD', {}).get('coverage', 0),
            class_coverage=counters.get('CLASS', {}).get('coverage', 0),
            total_lines=counters.get('LINE', {}).get('total', 0),
            covered_lines=counters.get('LINE', {}).get('covered', 0),
            total_branches=counters.get('BRANCH', {}).get('total', 0),
            covered_branches=counters.get('BRANCH', {}).get('covered', 0)
        )
    
    def _parse_mutation_report(self, xml_file: Path) -> MutationMetrics:
        """Parse PITest mutation XML report"""
        if not xml_file.exists():
            logger.warning(f"Mutation report not found: {xml_file}")
            return MutationMetrics(
                mutation_score=0,
                total_mutants=0,
                killed_mutants=0,
                survived_mutants=0,
                timeout_mutants=0,
                equivalent_mutants=0
            )
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        killed = 0
        survived = 0
        timeout = 0
        equivalent = 0
        
        for mutation in root.findall('.//mutation'):
            status = mutation.get('status', '')
            if status == 'KILLED':
                killed += 1
            elif status == 'SURVIVED':
                survived += 1
            elif status == 'TIMED_OUT':
                timeout += 1
            elif status == 'NO_COVERAGE' or status == 'MEMORY_ERROR':
                equivalent += 1
        
        total = killed + survived + timeout + equivalent
        score = (killed / total * 100) if total > 0 else 0
        
        return MutationMetrics(
            mutation_score=score,
            total_mutants=total,
            killed_mutants=killed,
            survived_mutants=survived,
            timeout_mutants=timeout,
            equivalent_mutants=equivalent
        )
    
    def _save_metrics(self, metrics: Any, filename: str) -> None:
        """Save metrics to JSON file"""
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {output_path}")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate summary report comparing results to paper claims.
        
        Returns:
            Dictionary with comparison results
        """
        logger.info("Generating summary report...")
        
        # Load all metrics
        metrics_files = {
            'coverage': 'coverage_metrics.json',
            'mutation': 'mutation_metrics.json',
            'performance': 'performance_metrics.json',
            'accuracy': 'accuracy_metrics.json',
            'acceptance': 'acceptance_metrics.json'
        }
        
        results = {}
        for key, filename in metrics_files.items():
            filepath = self.results_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    results[key] = json.load(f)
        
        # Compare to paper claims
        paper_claims = {
            'test_coverage': 89.0,
            'mutation_score': 84.0,
            'time_reduction': 55.0,
            'patch_accuracy': 91.0,
            'false_positive_rate': 8.0,
            'developer_acceptance': 82.0
        }
        
        comparison = {}
        
        if 'coverage' in results:
            comparison['test_coverage'] = {
                'paper_claim': paper_claims['test_coverage'],
                'measured': results['coverage']['line_coverage'],
                'difference': results['coverage']['line_coverage'] - paper_claims['test_coverage'],
                'meets_claim': results['coverage']['line_coverage'] >= paper_claims['test_coverage']
            }
        
        if 'mutation' in results:
            comparison['mutation_score'] = {
                'paper_claim': paper_claims['mutation_score'],
                'measured': results['mutation']['mutation_score'],
                'difference': results['mutation']['mutation_score'] - paper_claims['mutation_score'],
                'meets_claim': results['mutation']['mutation_score'] >= paper_claims['mutation_score']
            }
        
        if 'performance' in results and results['performance'].get('time_reduction'):
            comparison['time_reduction'] = {
                'paper_claim': paper_claims['time_reduction'],
                'measured': results['performance']['time_reduction'],
                'difference': results['performance']['time_reduction'] - paper_claims['time_reduction'],
                'meets_claim': results['performance']['time_reduction'] >= paper_claims['time_reduction']
            }
        
        if 'accuracy' in results:
            comparison['patch_accuracy'] = {
                'paper_claim': paper_claims['patch_accuracy'],
                'measured': results['accuracy']['accuracy'],
                'difference': results['accuracy']['accuracy'] - paper_claims['patch_accuracy'],
                'meets_claim': results['accuracy']['accuracy'] >= paper_claims['patch_accuracy']
            }
            
            comparison['false_positive_rate'] = {
                'paper_claim': paper_claims['false_positive_rate'],
                'measured': results['accuracy']['false_positive_rate'],
                'difference': paper_claims['false_positive_rate'] - results['accuracy']['false_positive_rate'],
                'meets_claim': results['accuracy']['false_positive_rate'] <= paper_claims['false_positive_rate']
            }
        
        if 'acceptance' in results:
            comparison['developer_acceptance'] = {
                'paper_claim': paper_claims['developer_acceptance'],
                'measured': results['acceptance']['acceptance_rate'],
                'difference': results['acceptance']['acceptance_rate'] - paper_claims['developer_acceptance'],
                'meets_claim': results['acceptance']['acceptance_rate'] >= paper_claims['developer_acceptance']
            }
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'paper_claims': paper_claims,
            'measured_results': results,
            'comparison': comparison
        }
        
        # Save summary
        summary_path = self.results_dir / 'summary_report.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary report saved to {summary_path}")
        
        return summary


if __name__ == "__main__":
    # Example usage
    evaluator = MetricsEvaluator()
    
    print("TestAgentX Metrics Evaluator")
    print("=" * 50)
    print("\nThis module provides tools to measure:")
    print("✓ Test coverage (Target: 89%)")
    print("✓ Mutation score (Target: 84%)")
    print("✓ Test generation time reduction (Target: 55%)")
    print("✓ Patch verification accuracy (Target: 91%)")
    print("✓ False positive rate (Target: 8%)")
    print("✓ Developer acceptance (Target: 82%)")
    print("\nSee evaluation/run_full_evaluation.py for complete evaluation pipeline")
