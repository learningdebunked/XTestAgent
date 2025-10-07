"""
Trace Analyzer for TestAgentX.

This module provides functionality for analyzing and comparing execution traces
between different versions of code to verify patch effectiveness.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Any, Optional
import difflib
import hashlib
from pathlib import Path
import json
import logging

from .patch_verification_agent import ExecutionTrace

@dataclass
class TraceDifference:
    """Represents differences between two execution traces."""
    test_id: str
    line_coverage_diff: Dict[str, List[int]]
    branch_coverage_diff: Dict[str, List[Tuple[int, int]]]
    method_call_diff: Dict[str, List[str]]
    execution_time_diff: float
    memory_usage_diff: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_id': self.test_id,
            'line_coverage_diff': {
                'added': sorted(self.line_coverage_diff.get('added', [])),
                'removed': sorted(self.line_coverage_diff.get('removed', [])),
                'common': sorted(self.line_coverage_diff.get('common', []))
            },
            'branch_coverage_diff': {
                'added': [f"{src}-{dst}" for src, dst in self.branch_coverage_diff.get('added', [])],
                'removed': [f"{src}-{dst}" for src, dst in self.branch_coverage_diff.get('removed', [])],
                'changed': self.branch_coverage_diff.get('changed', [])
            },
            'method_call_diff': {
                'added': sorted(self.method_call_diff.get('added', [])),
                'removed': sorted(self.method_call_diff.get('removed', [])),
                'common': sorted(self.method_call_diff.get('common', []))
            },
            'execution_time_diff': self.execution_time_diff,
            'memory_usage_diff': self.memory_usage_diff
        }

class TraceAnalyzer:
    """
    Analyzes and compares execution traces to determine patch effectiveness.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the TraceAnalyzer.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - similarity_threshold: Threshold for considering traces similar (0.0 to 1.0)
                - max_diff_lines: Maximum number of lines to show in diff output
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)
        self.max_diff_lines = self.config.get('max_diff_lines', 100)
        self.logger = logging.getLogger(__name__)
    
    def compare_traces(
        self,
        trace1: ExecutionTrace,
        trace2: ExecutionTrace,
        test_id: str = "unknown"
    ) -> TraceDifference:
        """Compare two execution traces and return their differences.
        
        Args:
            trace1: First execution trace (typically buggy version)
            trace2: Second execution trace (typically patched version)
            test_id: ID of the test case being compared
            
        Returns:
            TraceDifference object containing the differences
        """
        # Compare line coverage
        lines1 = set(trace1.covered_lines)
        lines2 = set(trace2.covered_lines)
        
        # Compare branch coverage
        branches1 = set(trace1.branch_coverage.items())
        branches2 = set(trace2.branch_coverage.items())
        
        # Compare method calls
        methods1 = set(trace1.method_calls)
        methods2 = set(trace2.method_calls)
        
        # Calculate branch differences
        branch_diff = self._compare_branches(trace1.branch_coverage, trace2.branch_coverage)
        
        return TraceDifference(
            test_id=test_id,
            line_coverage_diff={
                'added': sorted(lines2 - lines1),
                'removed': sorted(lines1 - lines2),
                'common': sorted(lines1 & lines2)
            },
            branch_coverage_diff=branch_diff,
            method_call_diff={
                'added': sorted(methods2 - methods1),
                'removed': sorted(methods1 - methods2),
                'common': sorted(methods1 & methods2)
            },
            execution_time_diff=trace2.execution_time - trace1.execution_time,
            memory_usage_diff=trace2.memory_usage - trace1.memory_usage
        )
    
    def calculate_patch_effectiveness(
        self,
        differences: Dict[str, TraceDifference],
        test_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate patch effectiveness scores based on trace differences.
        
        Args:
            differences: Dictionary mapping test IDs to their trace differences
            test_weights: Optional weights for different test cases
            
        Returns:
            Dictionary with various effectiveness metrics
        """
        if not differences:
            return {
                'overall_score': 0.0,
                'line_coverage_improvement': 0.0,
                'branch_coverage_improvement': 0.0,
                'method_coverage_improvement': 0.0,
                'performance_impact': 0.0,
                'memory_impact': 0.0
            }
        
        total_weight = 0.0
        weighted_scores = {
            'line_coverage': 0.0,
            'branch_coverage': 0.0,
            'method_coverage': 0.0,
            'performance': 0.0,
            'memory': 0.0
        }
        
        for test_id, diff in differences.items():
            weight = test_weights.get(test_id, 1.0) if test_weights else 1.0
            total_weight += weight
            
            # Calculate coverage improvements
            added_lines = len(diff.line_coverage_diff.get('added', []))
            removed_lines = len(diff.line_coverage_diff.get('removed', []))
            common_lines = len(diff.line_coverage_diff.get('common', []))
            total_lines = added_lines + removed_lines + common_lines
            
            line_score = (added_lines - (0.5 * removed_lines)) / max(1, total_lines)
            
            # Similar calculations for branches and methods
            added_branches = len(diff.branch_coverage_diff.get('added', []))
            removed_branches = len(diff.branch_coverage_diff.get('removed', []))
            changed_branches = len(diff.branch_coverage_diff.get('changed', []))
            total_branches = added_branches + removed_branches + changed_branches
            
            branch_score = (added_branches - (0.5 * removed_branches)) / max(1, total_branches)
            
            added_methods = len(diff.method_call_diff.get('added', []))
            removed_methods = len(diff.method_call_diff.get('removed', []))
            common_methods = len(diff.method_call_diff.get('common', []))
            total_methods = added_methods + removed_methods + common_methods
            
            method_score = (added_methods - (0.5 * removed_methods)) / max(1, total_methods)
            
            # Normalize scores to [0, 1] range
            line_score = max(0.0, min(1.0, 0.5 + (line_score / 2)))
            branch_score = max(0.0, min(1.0, 0.5 + (branch_score / 2)))
            method_score = max(0.0, min(1.0, 0.5 + (method_score / 2)))
            
            # Performance impact (negative is better)
            perf_impact = 1.0 - min(1.0, diff.execution_time_diff / max(1.0, diff.execution_time_diff + 1.0))
            
            # Memory impact (negative is better)
            mem_impact = 1.0 - min(1.0, diff.memory_usage_diff / max(1.0, diff.memory_usage_diff + 10.0))
            
            # Update weighted sums
            weighted_scores['line_coverage'] += weight * line_score
            weighted_scores['branch_coverage'] += weight * branch_score
            weighted_scores['method_coverage'] += weight * method_score
            weighted_scores['performance'] += weight * perf_impact
            weighted_scores['memory'] += weight * mem_impact
        
        # Calculate weighted averages
        results = {
            'overall_score': (
                weighted_scores['line_coverage'] * 0.3 +
                weighted_scores['branch_coverage'] * 0.3 +
                weighted_scores['method_coverage'] * 0.2 +
                weighted_scores['performance'] * 0.1 +
                weighted_scores['memory'] * 0.1
            ) / max(1, total_weight),
            'line_coverage_improvement': weighted_scores['line_coverage'] / max(1, total_weight),
            'branch_coverage_improvement': weighted_scores['branch_coverage'] / max(1, total_weight),
            'method_coverage_improvement': weighted_scores['method_coverage'] / max(1, total_weight),
            'performance_impact': weighted_scores['performance'] / max(1, total_weight),
            'memory_impact': weighted_scores['memory'] / max(1, total_weight)
        }
        
        return results
    
    def _compare_branches(
        self,
        branches1: Dict[Tuple[int, int], bool],
        branches2: Dict[Tuple[int, int], bool]
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Compare branch coverage between two versions."""
        branches1_set = set(branches1.items())
        branches2_set = set(branches2.items())
        
        # Find branches that exist in both versions but have different coverage
        changed = []
        common_branches = set(branches1.keys()) & set(branches2.keys())
        for branch in common_branches:
            if branches1[branch] != branches2[branch]:
                changed.append(branch)
        
        return {
            'added': [b for b, _ in (branches2_set - branches1_set)],
            'removed': [b for b, _ in (branches1_set - branches2_set)],
            'changed': changed
        }
    
    def generate_diff_report(
        self,
        diff: TraceDifference,
        source_code: Optional[Dict[int, str]] = None
    ) -> str:
        """Generate a human-readable diff report.
        
        Args:
            diff: TraceDifference object
            source_code: Optional mapping of line numbers to source code
            
        Returns:
            Formatted string with the diff report
        """
        report = [f"Test: {diff.test_id}", "=" * 80]
        
        # Line coverage differences
        report.append("\nLine Coverage Differences:")
        report.append(f"  Added lines: {len(diff.line_coverage_diff.get('added', []))}")
        report.append(f"  Removed lines: {len(diff.line_coverage_diff.get('removed', []))}")
        
        if source_code and diff.line_coverage_diff.get('added'):
            report.append("\nNewly covered lines:")
            for line in sorted(diff.line_coverage_diff['added'])[:self.max_diff_lines]:
                report.append(f"  {line}: {source_code.get(line, '')}")
        
        # Branch coverage differences
        report.append("\nBranch Coverage Differences:")
        report.append(f"  Added branches: {len(diff.branch_coverage_diff.get('added', []))}")
        report.append(f"  Removed branches: {len(diff.branch_coverage_diff.get('removed', []))}")
        report.append(f"  Changed branches: {len(diff.branch_coverage_diff.get('changed', []))}")
        
        # Method call differences
        report.append("\nMethod Call Differences:")
        report.append(f"  Added method calls: {len(diff.method_call_diff.get('added', []))}")
        report.append(f"  Removed method calls: {len(diff.method_call_diff.get('removed', []))}")
        
        # Performance impact
        report.append("\nPerformance Impact:")
        if diff.execution_time_diff > 0:
            report.append(f"  Slower by: {diff.execution_time_diff:.2f}s")
        else:
            report.append(f"  Faster by: {-diff.execution_time_diff:.2f}s")
            
        # Memory impact
        report.append("\nMemory Usage Impact:")
        if diff.memory_usage_diff > 0:
            report.append(f"  Increased by: {diff.memory_usage_diff:.2f} MB")
        else:
            report.append(f"  Decreased by: {-diff.memory_usage_diff:.2f} MB")
        
        return "\n".join(report)
    
    def save_traces_to_file(
        self,
        traces: Dict[str, ExecutionTrace],
        output_file: str
    ) -> None:
        """Save execution traces to a JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(
                    {test_id: trace.to_dict() for test_id, trace in traces.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            self.logger.error(f"Failed to save traces to {output_file}: {e}")
            raise
    
    @staticmethod
    def load_traces_from_file(input_file: str) -> Dict[str, ExecutionTrace]:
        """Load execution traces from a JSON file."""
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)
                return {
                    test_id: ExecutionTrace.from_dict(trace_data)
                    for test_id, trace_data in data.items()
                }
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load traces from {input_file}: {e}")
            raise
