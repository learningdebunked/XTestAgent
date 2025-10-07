"""
Visualization utilities for TestAgentX evaluation.

This module provides functions to generate various visualizations
for analyzing the performance of the test generation and validation pipeline.
"""

from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set the style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def plot_test_coverage(
    coverage_data: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot test coverage metrics over time or across different test cases.
    
    Args:
        coverage_data: Dictionary containing coverage data
                      (e.g., {'line_coverage': [...], 'branch_coverage': [...]})
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for metric, values in coverage_data.items():
        if values:  # Only plot if there's data
            ax.plot(values, label=metric.replace('_', ' ').title())
    
    ax.set_xlabel('Test Case #')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Test Coverage Over Test Cases')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig

def plot_fault_detection(
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot fault detection metrics (precision, recall, F1).
    
    Args:
        metrics: Dictionary containing metrics (precision, recall, f1)
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    metrics = {k: v for k, v in metrics.items() 
              if k in ['precision', 'recall', 'f1']}
    
    if not metrics:
        raise ValueError("No valid metrics provided for plotting")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    x = np.arange(len(metrics))
    values = list(metrics.values())
    
    bars = ax.bar(x, values, color=sns.color_palette("viridis", len(metrics)))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    ax.set_xticks(x)
    ax.set_xticklabels([k.upper() for k in metrics.keys()])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.set_title('Fault Detection Performance')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig

def plot_execution_time(
    timings: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot execution time distribution across different components.
    
    Args:
        timings: Dictionary of component names to lists of execution times
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not timings:
        raise ValueError("No timing data provided")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to list of (component, time) pairs
    data = []
    for component, times in timings.items():
        data.extend([(component, t) for t in times])
    
    import pandas as pd
    df = pd.DataFrame(data, columns=['Component', 'Time (s)'])
    
    # Create boxplot
    sns.boxplot(data=df, x='Component', y='Time (s)', ax=ax)
    ax.set_title('Execution Time Distribution by Component')
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig

def plot_validation_confidence(
    confidence_scores: List[float],
    labels: Optional[List[Any]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot the distribution of validation confidence scores.
    
    Args:
        confidence_scores: List of confidence scores (0-1)
        labels: Optional list of labels for each score
        save_path: Optional path to save the figure
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    if not confidence_scores:
        raise ValueError("No confidence scores provided")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    sns.histplot(confidence_scores, bins=20, kde=True, ax=ax)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Validation Confidence Scores')
    ax.set_xlim(0, 1)
    
    # Add vertical line at the mean
    mean_confidence = np.mean(confidence_scores)
    ax.axvline(mean_confidence, color='r', linestyle='--', 
               label=f'Mean: {mean_confidence:.2f}')
    ax.legend()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig

def generate_dashboard(
    metrics: Dict[str, Any],
    output_dir: str = 'reports/figures',
    prefix: str = ''
) -> Dict[str, str]:
    """
    Generate a complete dashboard of visualizations.
    
    Args:
        metrics: Dictionary containing all metrics and data
        output_dir: Directory to save the figures
        prefix: Optional prefix for output filenames
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # 1. Coverage over time
    if 'coverage_over_time' in metrics:
        fig_path = f"{output_dir}/{prefix}coverage_over_time.png"
        plot_test_coverage(
            metrics['coverage_over_time'],
            save_path=fig_path,
            show=False
        )
        figures['coverage'] = fig_path
    
    # 2. Fault detection metrics
    if all(m in metrics for m in ['precision', 'recall', 'f1']):
        fig_path = f"{output_dir}/{prefix}fault_detection.png"
        plot_fault_detection(
            {k: metrics[k] for k in ['precision', 'recall', 'f1']},
            save_path=fig_path,
            show=False
        )
        figures['fault_detection'] = fig_path
    
    # 3. Execution time
    if 'component_timings' in metrics:
        fig_path = f"{output_dir}/{prefix}execution_time.png"
        plot_execution_time(
            metrics['component_timings'],
            save_path=fig_path,
            show=False
        )
        figures['execution_time'] = fig_path
    
    # 4. Validation confidence
    if 'confidence_scores' in metrics:
        fig_path = f"{output_dir}/{prefix}validation_confidence.png"
        plot_validation_confidence(
            metrics['confidence_scores'],
            save_path=fig_path,
            show=False
        )
        figures['validation_confidence'] = fig_path
    
    return figures
