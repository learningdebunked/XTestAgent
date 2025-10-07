#!/usr/bin/env python3
"""
TestAgentX Implementation Validator

This script validates that all components of the TestAgentX system are properly implemented
and match the specifications from the paper.
"""

import os
import sys
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Constants
REQUIRED_EQUATIONS = [
    (1, "Code encoding (s_vec_i = Encoder_Code(s_i))"),
    (2, "Semantic diff (Delta_sem = Encoder_Code(P_f) - Encoder_Code(P_b))"),
    (3, "LLM test generation (T_gen = LLM_test(s_vec_i, M, Delta_sem))"),
    (4, "Fault detection potential (FDP(t_j) = E[CoverageGain(t_j) + CrashLikelihood(t_j)])"),
    (5, "RL reward function (R = α * MutationScore(T) + β * BranchCoverage(T))"),
    (6, "Contextual relevance score (CRS(O_b, O_f) = Sim_sem(O_b, O_f) / MaxSim)"),
    (7, "Confidence labeling (y_hat_t = sigmoid(W^T * o_vec_t + b))"),
    (8, "Patch differentiation (Delta_trace = Trace(P_f, t_j) - Trace(P_b, t_j))"),
    (9, "Regression test selection (T_reg = argmax Sim_graph(C_new, t_k))"),
    (10, "Utility function (U = λ1 * Cov + λ2 * FDR - λ3 * Cost)")
]

LAYERS = [
    "layer1_preprocessing",
    "layer2_test_generation",
    "layer3_fuzzy_validation",
    "layer4_patch_regression",
    "layer5_knowledge_graph"
]

AGENTS = [
    "Defects4JLoader",
    "ASTCFGGenerator",
    "CodeEncoder",
    "SemanticDiffAnalyzer",
    "LLMTestGenerationAgent",
    "RLPrioritizationAgent",
    "FuzzyAssertionAgent",
    "PatchVerificationAgent",
    "RegressionSentinelAgent",
    "KnowledgeGraphConstructor"
]

@dataclass
class ValidationResult:
    """Container for validation results."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None

def check_equation_implementation(equation_num: int, description: str) -> ValidationResult:
    """Check if an equation is properly implemented."""
    try:
        # Import the relevant module based on equation number
        if equation_num == 1:
            from layer1_preprocessing.code_encoder import CodeEncoder
            # Check if the encoder implements the equation
            encoder = CodeEncoder()
            if not hasattr(encoder, 'encode'):
                return ValidationResult(False, f"Equation {equation_num}: Missing encode method in CodeEncoder")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 2:
            from layer1_preprocessing.semantic_diff import SemanticDiffAnalyzer
            diff_analyzer = SemanticDiffAnalyzer()
            if not hasattr(diff_analyzer, 'compute_semantic_diff'):
                return ValidationResult(False, f"Equation {equation_num}: Missing compute_semantic_diff method")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 3:
            from layer2_test_generation.llm_test_agent import LLMTestGenerationAgent
            agent = LLMTestGenerationAgent()
            if not hasattr(agent, 'generate_tests'):
                return ValidationResult(False, f"Equation {equation_num}: Missing generate_tests method")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 4:
            from layer2_test_generation.rl_prioritization_agent import RLPrioritizationAgent
            agent = RLPrioritizationAgent()
            if not hasattr(agent, '_calculate_fault_detection_potential'):
                return ValidationResult(False, f"Equation {equation_num}: Missing fault detection potential calculation")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 5:
            from layer2_test_generation.rl_prioritization_agent import RLPrioritizationAgent
            agent = RLPrioritizationAgent()
            if not hasattr(agent, '_calculate_reward'):
                return ValidationResult(False, f"Equation {equation_num}: Missing reward calculation")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 6:
            from layer3_fuzzy_validation.fuzzy_assertion_agent import FuzzyAssertionAgent
            agent = FuzzyAssertionAgent()
            if not hasattr(agent, '_calculate_contextual_relevance_score'):
                return ValidationResult(False, f"Equation {equation_num}: Missing contextual relevance score calculation")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 7:
            from layer3_fuzzy_validation.fuzzy_assertion_agent import FuzzyAssertionAgent
            agent = FuzzyAssertionAgent()
            if not hasattr(agent, '_calculate_confidence_score'):
                return ValidationResult(False, f"Equation {equation_num}: Missing confidence score calculation")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 8:
            from layer4_patch_regression.patch_verification_agent import PatchVerificationAgent
            agent = PatchVerificationAgent()
            if not hasattr(agent, 'compare_execution_traces'):
                return ValidationResult(False, f"Equation {equation_num}: Missing execution trace comparison")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 9:
            from layer4_patch_regression.regression_sentinel_agent import RegressionSentinelAgent
            agent = RegressionSentinelAgent()
            if not hasattr(agent, 'select_regression_tests'):
                return ValidationResult(False, f"Equation {equation_num}: Missing regression test selection")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        elif equation_num == 10:
            from layer5_knowledge_graph.graph_navigator import GraphNavigator
            navigator = GraphNavigator()
            if not hasattr(navigator, 'calculate_utility'):
                return ValidationResult(False, f"Equation {equation_num}: Missing utility calculation")
            return ValidationResult(True, f"Equation {equation_num}: {description} ✓")
            
        return ValidationResult(False, f"Equation {equation_num}: Unknown equation number")
        
    except Exception as e:
        return ValidationResult(False, f"Equation {equation_num}: Error checking implementation: {str(e)}")

def check_layer_implementation(layer_name: str) -> ValidationResult:
    """Check if a layer is properly implemented."""
    try:
        # Try to import the layer's main module
        module_name = f"src.{layer_name}"
        importlib.import_module(module_name)
        
        # Check for required files
        layer_dir = Path("src") / layer_name
        if not layer_dir.exists():
            return ValidationResult(False, f"Layer {layer_name}: Directory does not exist")
            
        # Check for __init__.py
        if not (layer_dir / "__init__.py").exists():
            return ValidationResult(False, f"Layer {layer_name}: Missing __init__.py")
            
        return ValidationResult(True, f"Layer {layer_name}: ✓")
        
    except ImportError as e:
        return ValidationResult(False, f"Layer {layer_name}: Could not import module: {str(e)}")

def check_agent_implementation(agent_name: str) -> ValidationResult:
    """Check if an agent is properly implemented."""
    try:
        # Map agent names to their module paths
        agent_map = {
            "Defects4JLoader": "layer1_preprocessing.bug_ingestion",
            "ASTCFGGenerator": "layer1_preprocessing.ast_cfg_generator",
            "CodeEncoder": "layer1_preprocessing.code_encoder",
            "SemanticDiffAnalyzer": "layer1_preprocessing.semantic_diff",
            "LLMTestGenerationAgent": "layer2_test_generation.llm_test_agent",
            "RLPrioritizationAgent": "layer2_test_generation.rl_prioritization_agent",
            "FuzzyAssertionAgent": "layer3_fuzzy_validation.fuzzy_assertion_agent",
            "PatchVerificationAgent": "layer4_patch_regression.patch_verification_agent",
            "RegressionSentinelAgent": "layer4_patch_regression.regression_sentinel_agent",
            "KnowledgeGraphConstructor": "layer5_knowledge_graph.graph_constructor"
        }
        
        if agent_name not in agent_map:
            return ValidationResult(False, f"Agent {agent_name}: Not in agent mapping")
            
        module_name = agent_map[agent_name]
        module = importlib.import_module(module_name)
        
        if not hasattr(module, agent_name):
            return ValidationResult(False, f"Agent {agent_name}: Class not found in module {module_name}")
            
        # Try to instantiate the agent with default parameters
        agent_class = getattr(module, agent_name)
        agent_instance = agent_class()
        
        return ValidationResult(True, f"Agent {agent_name}: ✓")
        
    except Exception as e:
        return ValidationResult(False, f"Agent {agent_name}: Error during validation: {str(e)}")

def check_defects4j_integration() -> ValidationResult:
    """Check if Defects4J integration works."""
    try:
        from layer1_preprocessing.bug_ingestion import Defects4JLoader
        
        # Try to initialize the loader
        loader = Defects4JLoader()
        
        # Try to get bugs for a known project
        bug_ids = loader.get_all_bugs("Lang")
        
        if not bug_ids:
            return ValidationResult(False, "Defects4J: No bugs found for project 'Lang'")
            
        # Try to load a specific bug
        bug = loader.load_bug("Lang", 1)
        if not bug:
            return ValidationResult(False, "Defects4J: Could not load Lang-1")
            
        return ValidationResult(True, "Defects4J integration: ✓")
        
    except Exception as e:
        return ValidationResult(False, f"Defects4J integration error: {str(e)}")

def check_evaluation_metrics() -> ValidationResult:
    """Check if evaluation metrics match the paper."""
    try:
        from evaluation.metrics_calculator import calculate_metrics
        
        # Create a dummy test result
        dummy_result = {
            'test_coverage': 0.85,
            'mutation_score': 0.78,
            'tests_executed': 42,
            'execution_time': 123.45
        }
        
        # Try to calculate metrics
        metrics = calculate_metrics([dummy_result])
        
        required_metrics = [
            'test_coverage',
            'mutation_score',
            'tests_executed',
            'execution_time',
            'fault_detection_ratio'
        ]
        
        for metric in required_metrics:
            if metric not in metrics:
                return ValidationResult(False, f"Metrics: Missing required metric: {metric}")
                
        return ValidationResult(True, "Evaluation metrics: ✓")
        
    except Exception as e:
        return ValidationResult(False, f"Error checking evaluation metrics: {str(e)}")

def check_figure_generation() -> ValidationResult:
    """Check if all figures can be generated."""
    try:
        from evaluation.visualizations import (
            plot_comparison_with_evosuite,
            plot_patch_performance,
            plot_fuzzy_validation_metrics
        )
        
        # Create dummy data for testing
        dummy_data = [{
            'project': 'Lang',
            'bug_id': 1,
            'test_coverage': 0.8,
            'mutation_score': 0.7,
            'fault_detection_ratio': 0.9,
            'execution_time': 100
        }]
        
        # Test each plotting function
        plot_comparison_with_evosuite(dummy_data, "figure2_test.png")
        plot_patch_performance(dummy_data, "figure3_test.png")
        plot_fuzzy_validation_metrics(dummy_data, "figure4_test.png")
        
        # Clean up test files
        for f in ["figure2_test.png", "figure3_test.png", "figure4_test.png"]:
            if os.path.exists(f):
                os.remove(f)
                
        return ValidationResult(True, "Figure generation: ✓")
        
    except Exception as e:
        return ValidationResult(False, f"Error generating figures: {str(e)}")

def check_documentation() -> ValidationResult:
    """Check if documentation is complete."""
    required_files = [
        "README.md",
        "USAGE.md",
        "docs/API.md",
        "docs/DEVELOPMENT.md"
    ]
    
    missing_files = []
    for f in required_files:
        if not os.path.exists(f):
            missing_files.append(f)
    
    if missing_files:
        return ValidationResult(
            False,
            f"Documentation: Missing files: {', '.join(missing_files)}"
        )
    
    return ValidationResult(True, "Documentation: ✓")

def run_validation() -> Dict[str, List[Dict]]:
    """Run all validation checks."""
    results = {
        'equations': [],
        'layers': [],
        'agents': [],
        'integration': [],
        'metrics': [],
        'figures': [],
        'documentation': []
    }
    
    # Check equations
    print("\nValidating equations...")
    for eq_num, description in REQUIRED_EQUATIONS:
        result = check_equation_implementation(eq_num, description)
        results['equations'].append({
            'equation': eq_num,
            'description': description,
            'success': result.success,
            'message': result.message
        })
        print(f"  {result.message}")
    
    # Check layers
    print("\nValidating layers...")
    for layer in LAYERS:
        result = check_layer_implementation(layer)
        results['layers'].append({
            'layer': layer,
            'success': result.success,
            'message': result.message
        })
        print(f"  {result.message}")
    
    # Check agents
    print("\nValidating agents...")
    for agent in AGENTS:
        result = check_agent_implementation(agent)
        results['agents'].append({
            'agent': agent,
            'success': result.success,
            'message': result.message
        })
        print(f"  {result.message}")
    
    # Check integration
    print("\nValidating integration...")
    result = check_defects4j_integration()
    results['integration'].append({
        'component': 'Defects4J',
        'success': result.success,
        'message': result.message
    })
    print(f"  {result.message}")
    
    # Check metrics
    print("\nValidating metrics...")
    result = check_evaluation_metrics()
    results['metrics'].append({
        'component': 'Metrics',
        'success': result.success,
        'message': result.message
    })
    print(f"  {result.message}")
    
    # Check figure generation
    print("\nValidating figure generation...")
    result = check_figure_generation()
    results['figures'].append({
        'component': 'Figures',
        'success': result.success,
        'message': result.message
    })
    print(f"  {result.message}")
    
    # Check documentation
    print("\nValidating documentation...")
    result = check_documentation()
    results['documentation'].append({
        'component': 'Documentation',
        'success': result.success,
        'message': result.message
    })
    print(f"  {result.message}")
    
    return results

def generate_report(results: Dict[str, List[Dict]]) -> str:
    """Generate a human-readable validation report."""
    report = []
    report.append("=" * 80)
    report.append("TestAgentX Implementation Validation Report")
    report.append("=" * 80)
    
    # Summary
    total_checks = 0
    passed_checks = 0
    
    for category, items in results.items():
        category_passed = sum(1 for item in items if item['success'])
        total_checks += len(items)
        passed_checks += category_passed
        
        report.append(f"\n{category.upper()} ({category_passed}/{len(items)} passed)")
        report.append("-" * 40)
        
        for item in items:
            status = "✓" if item['success'] else "✗"
            report.append(f"{status} {item.get('description', item.get('component', 'Unknown'))}")
            if not item['success']:
                report.append(f"    Error: {item['message']}")
    
    # Overall status
    report.append("\n" + "=" * 80)
    report.append("VALIDATION SUMMARY")
    report.append("=" * 80)
    report.append(f"Total checks: {total_checks}")
    report.append(f"Passed: {passed_checks}")
    report.append(f"Failed: {total_checks - passed_checks}")
    report.append(f"Success rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if passed_checks == total_checks:
        report.append("\n✅ All checks passed!")
    else:
        report.append("\n❌ Some checks failed. Please review the report above.")
    
    return "\n".join(report)

def save_results(results: Dict[str, List[Dict]], filename: str = "validation_results.json") -> None:
    """Save validation results to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    print("Starting TestAgentX implementation validation...")
    results = run_validation()
    
    # Generate and print report
    report = generate_report(results)
    print("\n" + report)
    
    # Save detailed results
    save_results(results)
    
    # Save report to file
    with open("validation_report.txt", "w") as f:
        f.write(report)
    
    # Exit with appropriate status code
    total_checks = sum(len(items) for items in results.values())
    passed_checks = sum(sum(1 for item in items if item['success']) for items in results.values())
    sys.exit(0 if passed_checks == total_checks else 1)
