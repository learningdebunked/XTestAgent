"""
Metrics calculation for TestAgentX evaluation.

This module provides functions to calculate various metrics for evaluating
the performance of the test generation and validation pipeline.
"""
from typing import Dict, List, Tuple, Any
import numpy as np
from dataclasses import dataclass

@dataclass
class TestMetrics:
    """Class to hold test generation and validation metrics."""
    # Test generation metrics
    num_tests_generated: int = 0
    test_generation_time: float = 0.0
    
    # Coverage metrics
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    
    # Fault detection
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Validation metrics
    validation_accuracy: float = 0.0
    validation_precision: float = 0.0
    validation_recall: float = 0.0
    validation_f1: float = 0.0
    
    # Performance metrics
    execution_time: float = 0.0
    memory_usage: float = 0.0  # in MB

def calculate_metrics(
    test_results: List[Dict[str, Any]],
    ground_truth: Dict[str, Any]
) -> TestMetrics:
    """
    Calculate evaluation metrics from test results and ground truth.
    
    Args:
        test_results: List of test results from the pipeline
        ground_truth: Ground truth data for the test cases
        
    Returns:
        TestMetrics: Object containing all calculated metrics
    """
    metrics = TestMetrics()
    
    # Basic test generation metrics
    metrics.num_tests_generated = len(test_results)
    metrics.test_generation_time = sum(t.get('generation_time', 0) for t in test_results)
    
    # Calculate coverage metrics (simplified example)
    total_lines = ground_truth.get('total_lines', 1)
    covered_lines = sum(t.get('lines_covered', 0) for t in test_results)
    metrics.line_coverage = min(covered_lines / total_lines, 1.0) if total_lines > 0 else 0.0
    
    # Calculate fault detection metrics
    for test in test_results:
        test_id = test.get('test_id', '')
        is_faulty = ground_truth.get('faulty_tests', {}).get(test_id, False)
        is_detected = test.get('detected_issue', False)
        
        if is_faulty and is_detected:
            metrics.true_positives += 1
        elif is_faulty and not is_detected:
            metrics.false_negatives += 1
        elif not is_faulty and is_detected:
            metrics.false_positives += 1
        else:
            metrics.true_negatives += 1
    
    # Calculate precision, recall, F1
    tp, fp, fn = metrics.true_positives, metrics.false_positives, metrics.false_negatives
    metrics.validation_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics.validation_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics.validation_f1 = 2 * (metrics.validation_precision * metrics.validation_recall) / \
                          (metrics.validation_precision + metrics.validation_recall) \
                          if (metrics.validation_precision + metrics.validation_recall) > 0 else 0.0
    
    # Performance metrics
    metrics.execution_time = sum(t.get('execution_time', 0) for t in test_results)
    metrics.memory_usage = max(t.get('memory_usage', 0) for t in test_results, default=0)
    
    return metrics

def aggregate_metrics(
    metrics_list: List[TestMetrics],
    weights: List[float] = None
) -> Dict[str, float]:
    """
    Aggregate multiple TestMetrics objects into a single set of metrics.
    
    Args:
        metrics_list: List of TestMetrics objects to aggregate
        weights: Optional list of weights for weighted averaging
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not metrics_list:
        return {}
        
    if weights is None:
        weights = [1.0] * len(metrics_list)
    
    # Initialize sums
    sums = {}
        if field.startswith('_'):
            continue
        if field in ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']:
            sums[field] = sum(getattr(m, field, 0) for m in metrics_list)
        else:
            values = [getattr(m, field, 0) for m in metrics_list]
            if all(isinstance(v, (int, float)) for v in values):
                weighted_sum = sum((w * v for w, v in zip(weights, values)))
                sums[field] = weighted_sum / sum(weights) if weights else 0.0
    
    # Calculate aggregated metrics
    tp = sums.get('true_positives', 0)
    fp = sums.get('false_positives', 0)
    fn = sums.get('false_negatives', 0)
{{ ... }}
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Create result dictionary
    result = {
        'num_tests_generated': int(round(sums.get('num_tests_generated', 0))),
        'test_generation_time': sums.get('test_generation_time', 0.0),
        'line_coverage': sums.get('line_coverage', 0.0),
        'branch_coverage': sums.get('branch_coverage', 0.0),
        'true_positives': int(round(tp)),
        'false_positives': int(round(fp)),
        'true_negatives': int(round(sums.get('true_negatives', 0))),
        'false_negatives': int(round(fn)),
        'validation_precision': precision,
        'validation_recall': recall,
        'validation_f1': f1,
        'execution_time': sums.get('execution_time', 0.0),
        'memory_usage': sums.get('memory_usage', 0.0)
    }
    
    return result
