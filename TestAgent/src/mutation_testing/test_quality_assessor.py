"""
Test Quality Assessor for TestAgentX

Assesses test suite quality using mutation testing results.
Implements Section 4.2 of the paper: "Mutation testing is used to assess
the quality of generated test suites."
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics

from .mutation_engine import MutationResult, Mutant
from .mutation_analyzer import MutationAnalyzer, MutationReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestQualityMetrics:
    """Metrics for test suite quality"""
    mutation_score: float
    test_effectiveness: float  # 0-100
    coverage_adequacy: float  # 0-100
    fault_detection_capability: float  # 0-100
    overall_quality_score: float  # 0-100
    grade: str  # A, B, C, D, F
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mutation_score': self.mutation_score,
            'test_effectiveness': self.test_effectiveness,
            'coverage_adequacy': self.coverage_adequacy,
            'fault_detection_capability': self.fault_detection_capability,
            'overall_quality_score': self.overall_quality_score,
            'grade': self.grade
        }


class TestQualityAssessor:
    """
    Assesses test suite quality using mutation testing.
    
    Provides comprehensive quality metrics:
    - Mutation score
    - Test effectiveness
    - Fault detection capability
    - Overall quality grade
    """
    
    def __init__(self):
        """Initialize test quality assessor"""
        self.analyzer = MutationAnalyzer()
        logger.info("TestQualityAssessor initialized")
    
    def assess_quality(self, mutation_result: MutationResult,
                      code_coverage: Optional[float] = None) -> TestQualityMetrics:
        """
        Assess test suite quality.
        
        Args:
            mutation_result: Results from mutation testing
            code_coverage: Optional code coverage percentage
            
        Returns:
            TestQualityMetrics with quality assessment
        """
        logger.info("Assessing test suite quality")
        
        # Mutation score (primary metric)
        mutation_score = mutation_result.mutation_score
        
        # Test effectiveness (based on kill rate and diversity)
        test_effectiveness = self._calculate_test_effectiveness(mutation_result)
        
        # Coverage adequacy (combination of code coverage and mutation score)
        coverage_adequacy = self._calculate_coverage_adequacy(
            mutation_score, code_coverage
        )
        
        # Fault detection capability
        fault_detection = self._calculate_fault_detection_capability(mutation_result)
        
        # Overall quality score (weighted average)
        overall_score = (
            mutation_score * 0.4 +
            test_effectiveness * 0.3 +
            coverage_adequacy * 0.2 +
            fault_detection * 0.1
        )
        
        # Assign grade
        grade = self._assign_grade(overall_score)
        
        metrics = TestQualityMetrics(
            mutation_score=mutation_score,
            test_effectiveness=test_effectiveness,
            coverage_adequacy=coverage_adequacy,
            fault_detection_capability=fault_detection,
            overall_quality_score=overall_score,
            grade=grade
        )
        
        logger.info(f"Quality assessment: {grade} ({overall_score:.1f}%)")
        
        return metrics
    
    def _calculate_test_effectiveness(self, result: MutationResult) -> float:
        """Calculate test effectiveness score"""
        if result.total_mutants == 0:
            return 0.0
        
        # Base effectiveness on mutation score
        base_score = result.mutation_score
        
        # Bonus for killing diverse mutation types
        operator_diversity = len(set(m.operator_type for m in result.mutants 
                                    if m.status == "KILLED"))
        total_operators = len(set(m.operator_type for m in result.mutants))
        diversity_bonus = (operator_diversity / total_operators * 10) if total_operators > 0 else 0
        
        # Penalty for timeouts and errors
        timeout_penalty = (result.timeout_mutants / result.total_mutants * 5)
        error_penalty = (result.error_mutants / result.total_mutants * 10)
        
        effectiveness = base_score + diversity_bonus - timeout_penalty - error_penalty
        
        return max(0, min(100, effectiveness))
    
    def _calculate_coverage_adequacy(self, mutation_score: float,
                                    code_coverage: Optional[float]) -> float:
        """Calculate coverage adequacy score"""
        if code_coverage is None:
            # Use mutation score as proxy
            return mutation_score
        
        # Combine code coverage and mutation score
        # Mutation score is more important (70/30 split)
        adequacy = mutation_score * 0.7 + code_coverage * 0.3
        
        return adequacy
    
    def _calculate_fault_detection_capability(self, result: MutationResult) -> float:
        """Calculate fault detection capability"""
        if result.total_mutants == 0:
            return 0.0
        
        # Base capability on killed mutants
        kill_rate = (result.killed_mutants / result.total_mutants) * 100
        
        # Bonus for killing high-severity mutants (e.g., boundary conditions)
        high_severity_killed = sum(1 for m in result.mutants 
                                  if m.status == "KILLED" and 
                                  m.operator_type.value in ['boundary_value_adjustment', 
                                                            'relational_operator_replacement'])
        high_severity_total = sum(1 for m in result.mutants 
                                 if m.operator_type.value in ['boundary_value_adjustment',
                                                              'relational_operator_replacement'])
        
        severity_bonus = (high_severity_killed / high_severity_total * 10) if high_severity_total > 0 else 0
        
        capability = kill_rate + severity_bonus
        
        return min(100, capability)
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade based on score"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def compare_test_suites(self,
                          suite1_result: MutationResult,
                          suite2_result: MutationResult) -> Dict[str, Any]:
        """
        Compare quality of two test suites.
        
        Args:
            suite1_result: Mutation results for first suite
            suite2_result: Mutation results for second suite
            
        Returns:
            Comparison results
        """
        logger.info("Comparing test suite quality")
        
        metrics1 = self.assess_quality(suite1_result)
        metrics2 = self.assess_quality(suite2_result)
        
        comparison = {
            'suite1': metrics1.to_dict(),
            'suite2': metrics2.to_dict(),
            'winner': 'suite1' if metrics1.overall_quality_score > metrics2.overall_quality_score else 'suite2',
            'score_difference': abs(metrics1.overall_quality_score - metrics2.overall_quality_score),
            'mutation_score_diff': metrics2.mutation_score - metrics1.mutation_score,
            'effectiveness_diff': metrics2.test_effectiveness - metrics1.test_effectiveness,
            'summary': self._generate_comparison_summary(metrics1, metrics2)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, metrics1: TestQualityMetrics,
                                    metrics2: TestQualityMetrics) -> str:
        """Generate comparison summary"""
        score_diff = metrics2.overall_quality_score - metrics1.overall_quality_score
        
        if score_diff > 10:
            return f"Suite 2 is significantly better (Grade {metrics2.grade} vs {metrics1.grade})"
        elif score_diff > 5:
            return f"Suite 2 is moderately better (Grade {metrics2.grade} vs {metrics1.grade})"
        elif score_diff > 0:
            return f"Suite 2 is slightly better (Grade {metrics2.grade} vs {metrics1.grade})"
        elif score_diff < -10:
            return f"Suite 1 is significantly better (Grade {metrics1.grade} vs {metrics2.grade})"
        elif score_diff < -5:
            return f"Suite 1 is moderately better (Grade {metrics1.grade} vs {metrics2.grade})"
        elif score_diff < 0:
            return f"Suite 1 is slightly better (Grade {metrics1.grade} vs {metrics2.grade})"
        else:
            return f"Both suites have similar quality (Grade {metrics1.grade})"
    
    def assess_generated_tests(self,
                              generated_tests: List[str],
                              source_file: str,
                              test_command: str,
                              language: str) -> TestQualityMetrics:
        """
        Assess quality of generated tests using mutation testing.
        
        Args:
            generated_tests: List of generated test code
            source_file: Source file being tested
            test_command: Command to run tests
            language: Programming language
            
        Returns:
            TestQualityMetrics for generated tests
        """
        logger.info(f"Assessing {len(generated_tests)} generated tests")
        
        from .mutation_engine import MutationEngine
        
        # Run mutation testing
        engine = MutationEngine()
        mutation_result = engine.run_mutation_testing(
            source_file=source_file,
            test_command=test_command,
            language=language
        )
        
        # Assess quality
        metrics = self.assess_quality(mutation_result)
        
        logger.info(f"Generated tests quality: {metrics.grade} ({metrics.overall_quality_score:.1f}%)")
        
        return metrics
    
    def recommend_improvements(self, metrics: TestQualityMetrics,
                             mutation_report: MutationReport) -> List[str]:
        """
        Recommend improvements based on quality metrics.
        
        Args:
            metrics: Test quality metrics
            mutation_report: Mutation analysis report
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        # Overall quality
        if metrics.overall_quality_score < 70:
            recommendations.append(
                f"Overall test quality is low (Grade {metrics.grade}). "
                "Significant improvements needed."
            )
        
        # Mutation score
        if metrics.mutation_score < 80:
            recommendations.append(
                f"Mutation score is {metrics.mutation_score:.1f}%. "
                "Target: 80%+. Add tests for survived mutants."
            )
        
        # Test effectiveness
        if metrics.test_effectiveness < 75:
            recommendations.append(
                f"Test effectiveness is {metrics.test_effectiveness:.1f}%. "
                "Improve test diversity and assertion strength."
            )
        
        # Coverage adequacy
        if metrics.coverage_adequacy < 85:
            recommendations.append(
                f"Coverage adequacy is {metrics.coverage_adequacy:.1f}%. "
                "Increase both code coverage and mutation score."
            )
        
        # Fault detection
        if metrics.fault_detection_capability < 80:
            recommendations.append(
                f"Fault detection capability is {metrics.fault_detection_capability:.1f}%. "
                "Add tests for boundary conditions and edge cases."
            )
        
        # Specific weak spots
        if mutation_report.weak_spots:
            recommendations.append(
                f"Focus on {len(mutation_report.weak_spots)} identified weak spots:"
            )
            for spot in mutation_report.weak_spots[:3]:
                recommendations.append(
                    f"  - {spot['file']}:{spot['line']} needs better test coverage"
                )
        
        if not recommendations:
            recommendations.append(
                f"Test suite quality is excellent (Grade {metrics.grade}). "
                "Continue maintaining high standards."
            )
        
        return recommendations
    
    def generate_quality_report(self, metrics: TestQualityMetrics,
                               output_path: str) -> None:
        """Generate quality assessment report"""
        report = f"""
TEST SUITE QUALITY REPORT
{'='*60}

Overall Grade: {metrics.grade}
Overall Score: {metrics.overall_quality_score:.1f}%

DETAILED METRICS
{'='*60}
Mutation Score:              {metrics.mutation_score:.1f}%
Test Effectiveness:          {metrics.test_effectiveness:.1f}%
Coverage Adequacy:           {metrics.coverage_adequacy:.1f}%
Fault Detection Capability:  {metrics.fault_detection_capability:.1f}%

INTERPRETATION
{'='*60}
"""
        
        if metrics.grade == 'A':
            report += "Excellent test suite quality. Tests are comprehensive and effective.\n"
        elif metrics.grade == 'B':
            report += "Good test suite quality. Minor improvements recommended.\n"
        elif metrics.grade == 'C':
            report += "Adequate test suite quality. Several improvements needed.\n"
        elif metrics.grade == 'D':
            report += "Below average test suite quality. Significant improvements required.\n"
        else:
            report += "Poor test suite quality. Major overhaul needed.\n"
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Quality report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    assessor = TestQualityAssessor()
    
    # Assess quality
    # metrics = assessor.assess_quality(mutation_result)
    # print(f"Quality Grade: {metrics.grade}")
    # print(f"Overall Score: {metrics.overall_quality_score:.1f}%")
