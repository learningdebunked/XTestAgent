"""
TestAgentX - Automated Test Generation and Validation Framework

This package provides a comprehensive framework for automated test generation and validation.
It includes multiple layers for different aspects of the testing process.
"""

__version__ = "0.1.0"
__author__ = "Your Name <your.email@example.com>"

# Import key components to make them available at package level
from .layer1_preprocessing import *
from .layer2_test_generation import *
from .layer3_fuzzy_validation import *
from .layer4_patch_regression import *
from .layer5_knowledge_graph import *
from .agents import *
from .explainability import *
from .utils import *

__all__ = [
    # Modules
    'agents',
    'explainability',
    'utils',
    
    # Layers
    'layer1_preprocessing',
    'layer2_test_generation',
    'layer3_fuzzy_validation',
    'layer4_patch_regression',
    'layer5_knowledge_graph',
    
    # Version
    '__version__',
    '__author__'
]
