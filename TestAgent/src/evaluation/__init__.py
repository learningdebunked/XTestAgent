"""
Evaluation module for TestAgentX.

This module provides functionality for evaluating the performance of the test generation
and validation pipeline, including metrics calculation and visualization.
"""

from .metrics_calculator import calculate_metrics, aggregate_metrics
from .visualizations import (
    plot_test_coverage,
    plot_fault_detection,
    plot_execution_time,
    plot_validation_confidence
)

__all__ = [
    'calculate_metrics',
    'aggregate_metrics',
    'plot_test_coverage',
    'plot_fault_detection',
    'plot_execution_time',
    'plot_validation_confidence'
]
