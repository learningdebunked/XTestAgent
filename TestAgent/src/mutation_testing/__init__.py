"""
Mutation Testing Module for TestAgentX

Implements mutation testing to evaluate test suite quality.
Based on Section 4.2 of the paper: "Mutation testing is used to assess
the quality of generated test suites."

This module provides:
- Mutation operators for Java and Python
- Mutation execution engine
- Test suite quality assessment
- Mutation score calculation
"""

from .mutation_operators import MutationOperator, MutationOperatorType
from .mutation_engine import MutationEngine, MutationResult
from .mutation_analyzer import MutationAnalyzer, MutationReport
from .test_quality_assessor import TestQualityAssessor, TestQualityMetrics

__all__ = [
    'MutationOperator',
    'MutationOperatorType',
    'MutationEngine',
    'MutationResult',
    'MutationAnalyzer',
    'MutationReport',
    'TestQualityAssessor',
    'TestQualityMetrics'
]
