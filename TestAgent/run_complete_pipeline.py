#!/usr/bin/env python3
"""
TestAgentX: Complete End-to-End Pipeline

This script reproduces all results and figures from the paper.
"""

import sys
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import all required components
from layer1_preprocessing.bug_ingestion import Defects4JLoader, BugInstance
from layer1_preprocessing.ast_cfg_generator import ASTCFGGenerator
from layer1_preprocessing.code_encoder import CodeEncoder
from layer1_preprocessing.semantic_diff import SemanticDiffAnalyzer
from layer2_test_generation.llm_test_agent import LLMTestGenerationAgent
from layer2_test_generation.rl_prioritization_agent import RLPrioritizationAgent
from layer3_fuzzy_validation.fuzzy_assertion_agent import FuzzyAssertionAgent
from layer4_patch_regression.patch_verification_agent import PatchVerificationAgent
from layer4_patch_regression.regression_sentinel_agent import RegressionSentinelAgent
from layer5_knowledge_graph.graph_constructor import KnowledgeGraphConstructor
from agents.agent_orchestrator import AgentOrchestrator, AgentConfig

# Constants
DEFAULT_PROJECTS = ["Lang", "Chart", "Time", "Math", "Closure", "Mockito"]
OUTPUT_DIR = Path("results")
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = Path("models")

@dataclass
class PipelineConfig:
    """Configuration for the pipeline execution."""
    projects: List[str] = None
    max_bugs: Optional[int] = None
    skip_existing: bool = True
    debug: bool = False
    output_dir: Path = OUTPUT_DIR
    figures_dir: Path = FIGURES_DIR
    models_dir: Path = MODELS_DIR

def setup_environment(config: PipelineConfig) -> None:
    """Set up the execution environment."""
    logger.info("Setting up environment...")
    
    # Create output directories
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    if config.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Output directory: {config.output_dir.absolute()}")
    logger.info(f"Figures directory: {config.figures_dir.absolute()}")
    logger.info(f"Models directory: {config.models_dir.absolute()}")

def initialize_agents(config: PipelineConfig) -> Dict[str, Any]:
    """Initialize all agents and components."""
    logger.info("Initializing agents and components...")
    
    # Initialize the orchestrator
    orchestrator = AgentOrchestrator()
    
    # Register all agents with their configurations
    agents_config = {
        'bug_loader': AgentConfig(
            agent_class=Defects4JLoader,
            config={'projects': config.projects}
        ),
        'ast_cfg_generator': AgentConfig(
            agent_class=ASTCFGGenerator,
            config={},
            dependencies=['bug_loader']
        ),
        'code_encoder': AgentConfig(
            agent_class=CodeEncoder,
            config={'model_name': 'microsoft/codebert-base'},
            dependencies=['ast_cfg_generator']
        ),
        'semantic_diff': AgentConfig(
            agent_class=SemanticDiffAnalyzer,
            config={},
            dependencies=['code_encoder']
        ),
        'llm_test_generator': AgentConfig(
            agent_class=LLMTestGenerationAgent,
            config={'model_name': 'gpt-4'},
            dependencies=['semantic_diff']
        ),
        'rl_prioritizer': AgentConfig(
            agent_class=RLPrioritizationAgent,
            config={'alpha': 0.7, 'beta': 0.3},
            dependencies=['llm_test_generator']
        ),
        'fuzzy_validator': AgentConfig(
            agent_class=FuzzyAssertionAgent,
            config={'threshold': 0.8},
            dependencies=['rl_prioritizer']
        ),
        'patch_verifier': AgentConfig(
            agent_class=PatchVerificationAgent,
            config={},
            dependencies=['fuzzy_validator']
        ),
        'regression_sentinel': AgentConfig(
            agent_class=RegressionSentinelAgent,
            config={},
            dependencies=['patch_verifier']
        ),
        'knowledge_graph': AgentConfig(
            agent_class=KnowledgeGraphConstructor,
            config={'db_path': 'data/knowledge_graph.db'},
            dependencies=['regression_sentinel']
        )
    }
    
    # Register all agents with the orchestrator
    for name, agent_config in agents_config.items():
        orchestrator.add_agent(name, agent_config)
    
    # Initialize the orchestrator
    orchestrator.initialize()
    
    return {
        'orchestrator': orchestrator,
        'agents': {name: orchestrator.get_agent(name) for name in agents_config}
    }

def load_bugs(loader: Defects4JLoader, config: PipelineConfig) -> List[BugInstance]:
    """Load bugs from Defects4J."""
    logger.info(f"Loading bugs for projects: {', '.join(config.projects)}")
    
    all_bugs = []
    for project in config.projects:
        try:
            bug_ids = loader.get_all_bugs(project)
            for bid in bug_ids[:config.max_bugs]:
                bug = loader.load_bug(project, bid)
                if bug:
                    all_bugs.append(bug)
            logger.info(f"Loaded {len(bug_ids)} bugs from {project}")
        except Exception as e:
            logger.error(f"Error loading bugs for {project}: {str(e)}")
    
    logger.info(f"Total bugs loaded: {len(all_bugs)}")
    return all_bugs

def run_pipeline(config: PipelineConfig) -> None:
    """Run the complete pipeline."""
    start_time = time.time()
    
    try:
        # Setup environment
        setup_environment(config)
        
        # Initialize agents
        components = initialize_agents(config)
        orchestrator = components['orchestrator']
        
        # Load bugs
        bug_loader = components['agents']['bug_loader']
        bugs = load_bugs(bug_loader, config)
        
        # Process each bug
        results = []
        for i, bug in enumerate(bugs, 1):
            logger.info(f"\nProcessing bug {i}/{len(bugs)}: {bug.project}-{bug.bug_id}")
            
            try:
                # Execute the workflow for this bug
                result = orchestrator.execute_workflow(
                    start_agent='ast_cfg_generator',
                    input_data=bug,
                    max_steps=100
                )
                
                # Record results
                if result and 'metrics' in result:
                    results.append({
                        'project': bug.project,
                        'bug_id': bug.bug_id,
                        **result['metrics']
                    })
                
                # Save intermediate results
                if i % 10 == 0:
                    save_results(results, config.output_dir / 'intermediate_results.csv')
            
            except Exception as e:
                logger.error(f"Error processing {bug.project}-{bug.bug_id}: {str(e)}", 
                            exc_info=config.debug)
        
        # Save final results
        save_results(results, config.output_dir / 'final_results.csv')
        
        # Generate visualizations
        generate_visualizations(results, config.figures_dir)
        
        # Save trained models
        save_models(components['agents'], config.models_dir)
        
        logger.info(f"\nPipeline completed in {(time.time() - start_time)/60:.2f} minutes")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=config.debug)
        raise

def save_results(results: List[Dict], output_path: Path) -> None:
    """Save results to a CSV file."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

def generate_visualizations(results: List[Dict], output_dir: Path) -> None:
    """Generate all visualizations."""
    try:
        logger.info("Generating visualizations...")
        
        # Import visualization functions
        from evaluation.visualizations import (
            plot_comparison_with_evosuite,
            plot_patch_performance,
            plot_fuzzy_validation_metrics
        )
        
        # Generate each figure
        plot_comparison_with_evosuite(results, output_dir / 'figure2_comparison.png')
        plot_patch_performance(results, output_dir / 'figure3_patch_performance.png')
        plot_fuzzy_validation_metrics(results, output_dir / 'figure4_fuzzy_validation.png')
        
        logger.info(f"Visualizations saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)

def save_models(agents: Dict[str, Any], output_dir: Path) -> None:
    """Save trained models."""
    logger.info("Saving trained models...")
    
    for name, agent in agents.items():
        if hasattr(agent, 'save_model'):
            try:
                agent_dir = output_dir / name
                agent_dir.mkdir(parents=True, exist_ok=True)
                agent.save_model(agent_dir)
                logger.debug(f"Saved model for {name}")
            except Exception as e:
                logger.warning(f"Could not save model for {name}: {str(e)}")
    
    logger.info(f"Models saved to {output_dir}")

def main():
    """Main entry point for the pipeline."""
    print("=" * 80)
    print("TestAgentX: Complete Pipeline Execution")
    print("=" * 80)
    
    # Configuration
    config = PipelineConfig(
        projects=DEFAULT_PROJECTS,
        max_bugs=10,  # Set to None to process all bugs
        debug=True,
        output_dir=OUTPUT_DIR,
        figures_dir=FIGURES_DIR,
        models_dir=MODELS_DIR
    )
    
    # Run the pipeline
    run_pipeline(config)
    
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"Results saved to: {config.output_dir.absolute()}")
    print(f"Figures saved to: {config.figures_dir.absolute()}")
    print(f"Models saved to: {config.models_dir.absolute()}")

if __name__ == "__main__":
    main()
