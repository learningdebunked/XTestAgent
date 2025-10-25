"""
Mutation Analyzer for TestAgentX

Analyzes mutation testing results to provide insights about test quality.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

from .mutation_engine import MutationResult, Mutant
from .mutation_operators import MutationOperatorType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MutationReport:
    """Comprehensive mutation testing report"""
    mutation_score: float
    total_mutants: int
    killed_mutants: int
    survived_mutants: int
    weak_spots: List[Dict[str, Any]] = field(default_factory=list)
    operator_effectiveness: Dict[str, float] = field(default_factory=dict)
    file_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mutation_score': self.mutation_score,
            'total_mutants': self.total_mutants,
            'killed_mutants': self.killed_mutants,
            'survived_mutants': self.survived_mutants,
            'weak_spots': self.weak_spots,
            'operator_effectiveness': self.operator_effectiveness,
            'file_scores': self.file_scores,
            'recommendations': self.recommendations
        }


class MutationAnalyzer:
    """
    Analyzes mutation testing results.
    
    Provides insights about:
    - Test suite quality
    - Weak spots in testing
    - Operator effectiveness
    - Recommendations for improvement
    """
    
    def __init__(self):
        """Initialize mutation analyzer"""
        logger.info("MutationAnalyzer initialized")
    
    def analyze(self, result: MutationResult) -> MutationReport:
        """
        Analyze mutation testing results.
        
        Args:
            result: MutationResult from mutation testing
            
        Returns:
            MutationReport with analysis
        """
        logger.info("Analyzing mutation testing results")
        
        # Identify weak spots (survived mutants)
        weak_spots = self._identify_weak_spots(result.mutants)
        
        # Analyze operator effectiveness
        operator_effectiveness = self._analyze_operator_effectiveness(result.mutants)
        
        # Calculate per-file scores
        file_scores = self._calculate_file_scores(result.mutants)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            result, weak_spots, operator_effectiveness
        )
        
        report = MutationReport(
            mutation_score=result.mutation_score,
            total_mutants=result.total_mutants,
            killed_mutants=result.killed_mutants,
            survived_mutants=result.survived_mutants,
            weak_spots=weak_spots,
            operator_effectiveness=operator_effectiveness,
            file_scores=file_scores,
            recommendations=recommendations
        )
        
        logger.info(f"Analysis complete: {len(weak_spots)} weak spots found")
        
        return report
    
    def _identify_weak_spots(self, mutants: List[Mutant]) -> List[Dict[str, Any]]:
        """Identify weak spots where mutants survived"""
        weak_spots = []
        
        # Group survived mutants by file and line
        survived_by_location = defaultdict(list)
        
        for mutant in mutants:
            if mutant.status == "SURVIVED":
                key = (mutant.file_path, mutant.line_number)
                survived_by_location[key].append(mutant)
        
        # Create weak spot entries
        for (file_path, line_number), mutant_list in survived_by_location.items():
            weak_spot = {
                'file': file_path,
                'line': line_number,
                'survived_mutants': len(mutant_list),
                'operators': [m.operator_name for m in mutant_list],
                'severity': 'HIGH' if len(mutant_list) > 2 else 'MEDIUM'
            }
            weak_spots.append(weak_spot)
        
        # Sort by number of survived mutants
        weak_spots.sort(key=lambda x: x['survived_mutants'], reverse=True)
        
        return weak_spots
    
    def _analyze_operator_effectiveness(self, mutants: List[Mutant]) -> Dict[str, float]:
        """Analyze effectiveness of each mutation operator"""
        operator_stats = defaultdict(lambda: {'killed': 0, 'total': 0})
        
        for mutant in mutants:
            operator = mutant.operator_type.value
            operator_stats[operator]['total'] += 1
            
            if mutant.status == "KILLED":
                operator_stats[operator]['killed'] += 1
        
        # Calculate kill rate for each operator
        effectiveness = {}
        for operator, stats in operator_stats.items():
            if stats['total'] > 0:
                kill_rate = (stats['killed'] / stats['total']) * 100
                effectiveness[operator] = kill_rate
        
        return effectiveness
    
    def _calculate_file_scores(self, mutants: List[Mutant]) -> Dict[str, float]:
        """Calculate mutation score per file"""
        file_stats = defaultdict(lambda: {'killed': 0, 'total': 0})
        
        for mutant in mutants:
            file_path = mutant.file_path
            file_stats[file_path]['total'] += 1
            
            if mutant.status == "KILLED":
                file_stats[file_path]['killed'] += 1
        
        # Calculate score for each file
        scores = {}
        for file_path, stats in file_stats.items():
            if stats['total'] > 0:
                score = (stats['killed'] / stats['total']) * 100
                scores[file_path] = score
        
        return scores
    
    def _generate_recommendations(self,
                                 result: MutationResult,
                                 weak_spots: List[Dict[str, Any]],
                                 operator_effectiveness: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving test suite"""
        recommendations = []
        
        # Overall mutation score
        if result.mutation_score < 60:
            recommendations.append(
                f"Low mutation score ({result.mutation_score:.1f}%). "
                "Consider adding more comprehensive tests."
            )
        elif result.mutation_score < 80:
            recommendations.append(
                f"Moderate mutation score ({result.mutation_score:.1f}%). "
                "Focus on testing edge cases and boundary conditions."
            )
        else:
            recommendations.append(
                f"Good mutation score ({result.mutation_score:.1f}%). "
                "Continue maintaining test quality."
            )
        
        # Weak spots
        if weak_spots:
            top_weak_spots = weak_spots[:3]
            recommendations.append(
                f"Found {len(weak_spots)} weak spots. Top concerns:"
            )
            for spot in top_weak_spots:
                recommendations.append(
                    f"  - {spot['file']}:{spot['line']} has {spot['survived_mutants']} "
                    f"survived mutants. Add tests for: {', '.join(spot['operators'][:2])}"
                )
        
        # Operator effectiveness
        low_effectiveness = {op: rate for op, rate in operator_effectiveness.items() 
                           if rate < 50}
        if low_effectiveness:
            recommendations.append(
                "Low kill rates for certain mutation types:"
            )
            for operator, rate in list(low_effectiveness.items())[:3]:
                recommendations.append(
                    f"  - {operator}: {rate:.1f}% kill rate. "
                    f"Add tests targeting this mutation type."
                )
        
        # Survived mutants
        if result.survived_mutants > result.total_mutants * 0.3:
            recommendations.append(
                f"High number of survived mutants ({result.survived_mutants}). "
                "Review test assertions and add negative test cases."
            )
        
        return recommendations
    
    def compare_results(self, 
                       before: MutationResult,
                       after: MutationResult) -> Dict[str, Any]:
        """
        Compare two mutation testing results.
        
        Args:
            before: Results before changes
            after: Results after changes
            
        Returns:
            Dictionary with comparison metrics
        """
        score_change = after.mutation_score - before.mutation_score
        killed_change = after.killed_mutants - before.killed_mutants
        
        comparison = {
            'score_change': score_change,
            'score_before': before.mutation_score,
            'score_after': after.mutation_score,
            'killed_change': killed_change,
            'killed_before': before.killed_mutants,
            'killed_after': after.killed_mutants,
            'improvement': score_change > 0,
            'summary': self._generate_comparison_summary(score_change, killed_change)
        }
        
        return comparison
    
    def _generate_comparison_summary(self, score_change: float, 
                                    killed_change: int) -> str:
        """Generate summary of comparison"""
        if score_change > 5:
            return f"Significant improvement: +{score_change:.1f}% mutation score"
        elif score_change > 0:
            return f"Slight improvement: +{score_change:.1f}% mutation score"
        elif score_change < -5:
            return f"Significant regression: {score_change:.1f}% mutation score"
        elif score_change < 0:
            return f"Slight regression: {score_change:.1f}% mutation score"
        else:
            return "No significant change in mutation score"
    
    def generate_html_report(self, report: MutationReport, 
                            output_path: str) -> None:
        """Generate HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mutation Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                  background: #f0f0f0; border-radius: 5px; }}
        .score {{ font-size: 2em; font-weight: bold; }}
        .good {{ color: green; }}
        .medium {{ color: orange; }}
        .bad {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .weak-spot {{ background-color: #ffebee; }}
    </style>
</head>
<body>
    <h1>Mutation Testing Report</h1>
    
    <div class="metrics">
        <div class="metric">
            <div>Mutation Score</div>
            <div class="score {'good' if report.mutation_score >= 80 else 'medium' if report.mutation_score >= 60 else 'bad'}">
                {report.mutation_score:.1f}%
            </div>
        </div>
        <div class="metric">
            <div>Total Mutants</div>
            <div class="score">{report.total_mutants}</div>
        </div>
        <div class="metric">
            <div>Killed</div>
            <div class="score good">{report.killed_mutants}</div>
        </div>
        <div class="metric">
            <div>Survived</div>
            <div class="score bad">{report.survived_mutants}</div>
        </div>
    </div>
    
    <h2>Weak Spots</h2>
    <table>
        <tr>
            <th>File</th>
            <th>Line</th>
            <th>Survived Mutants</th>
            <th>Severity</th>
        </tr>
        {''.join(f'''
        <tr class="weak-spot">
            <td>{spot['file']}</td>
            <td>{spot['line']}</td>
            <td>{spot['survived_mutants']}</td>
            <td>{spot['severity']}</td>
        </tr>
        ''' for spot in report.weak_spots[:10])}
    </table>
    
    <h2>Recommendations</h2>
    <ul>
        {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
    </ul>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"HTML report generated: {output_path}")


if __name__ == "__main__":
    # Example usage
    from .mutation_engine import MutationEngine
    
    analyzer = MutationAnalyzer()
    
    # Analyze results
    # result = engine.run_mutation_testing(...)
    # report = analyzer.analyze(result)
    # print(f"Mutation score: {report.mutation_score:.2f}%")
    # print(f"Weak spots: {len(report.weak_spots)}")
