"""
Implements the fuzzy validation layer for test output validation.

Implements:
- Equation (6): CRS(O_b, O_f) = Sim_sem(O_b, O_f) / MaxSim
- Equation (7): y_hat_t = sigmoid(W^T * o_vec_t + b)
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, util
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    confidence: float
    semantic_similarity: float
    context_similarity: float
    metadata: Dict[str, Any] = None

class FuzzyAssertionAgent:
    """
    Implements fuzzy validation of test outputs using semantic similarity and learned scoring.
    """
    
    def __init__(self, 
                model_name: str = 'all-mpnet-base-v2',
                device: str = None,
                threshold: float = 0.7,
                max_sim: float = 1.0):
        """
        Initialize the fuzzy assertion agent.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run the model on ('cuda' or 'cpu')
            threshold: Threshold for validation (default: 0.7)
            max_sim: Maximum possible similarity value (for normalization)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self.max_sim = max_sim
        
        # Initialize the sentence transformer model
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Initialize the scoring layer (Equation 7)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.scoring_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        logger.info(f"FuzzyAssertionAgent initialized on device: {self.device}")
    
    def validate_output(self, 
                       output_buggy: str, 
                       output_fixed: str,
                       context: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate if the output from the fixed version is semantically 
        consistent with the buggy version's output.
        
        Implements Equations (6) and (7) from the paper.
        
        Args:
            output_buggy: Output from the buggy version
            output_fixed: Output from the fixed version
            context: Additional context for validation
            
        Returns:
            ValidationResult with validation decision and confidence
        """
        try:
            # Encode the outputs
            embeddings = self.model.encode([output_buggy, output_fixed], 
                                         convert_to_tensor=True,
                                         device=self.device)
            
            # Calculate semantic similarity (Equation 6)
            sim_sem = self._calculate_semantic_similarity(embeddings[0], embeddings[1])
            
            # Calculate contextual relevance score (CRS)
            crs = self._calculate_contextual_relevance_score(sim_sem)
            
            # Calculate final confidence score (Equation 7)
            confidence = self._calculate_confidence_score(embeddings[1], context)
            
            # Make validation decision
            is_valid = confidence >= self.threshold
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=float(confidence),
                semantic_similarity=float(sim_sem),
                context_similarity=float(crs),
                metadata={
                    'output_buggy': output_buggy,
                    'output_fixed': output_fixed,
                    'threshold': self.threshold
                }
            )
            
        except Exception as e:
            logger.error(f"Error in validate_output: {str(e)}", exc_info=True)
            # Return invalid result with minimum confidence on error
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                semantic_similarity=0.0,
                context_similarity=0.0,
                metadata={
                    'error': str(e),
                    'output_buggy': output_buggy,
                    'output_fixed': output_fixed
                }
            )
    
    def _calculate_semantic_similarity(self, 
                                     emb_buggy: torch.Tensor,
                                     emb_fixed: torch.Tensor) -> float:
        """
        Calculate semantic similarity between two embeddings.
        
        Args:
            emb_buggy: Embedding of the buggy output
            emb_fixed: Embedding of the fixed output
            
        Returns:
            Semantic similarity score (0.0 to 1.0)
        ""
        # Use cosine similarity
        sim = util.pytorch_cos_sim(emb_buggy.unsqueeze(0), 
                                  emb_fixed.unsqueeze(0))
        return sim.item()
    
    def _calculate_contextual_relevance_score(self, 
                                            semantic_similarity: float) -> float:
        """
        Calculate Contextual Relevance Score (CRS) using Equation (6).
        
        CRS(O_b, O_f) = Sim_sem(O_b, O_f) / MaxSim
        
        Args:
            semantic_similarity: Semantic similarity score (0.0 to 1.0)
            
        Returns:
            Contextual relevance score (0.0 to 1.0)
        """
        return semantic_similarity / self.max_sim
    
    def _calculate_confidence_score(self,
                                  embedding: torch.Tensor,
                                  context: Dict[str, Any] = None) -> float:
        """
        Calculate confidence score using Equation (7).
        
        y_hat_t = sigmoid(W^T * o_vec_t + b)
        
        Args:
            embedding: Output embedding to score
            context: Optional context for the scoring
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        with torch.no_grad():
            # Ensure embedding is on the right device and has batch dimension
            embedding = embedding.unsqueeze(0).to(self.device)
            
            # Apply learned scoring (Equation 7)
            confidence = self.scoring_layer(embedding)
            
            return confidence.item()
    
    def save_model(self, path: str) -> None:
        """
        Save the model and scoring layer to disk.
        
        Args:
            path: Directory to save the model to
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save the sentence transformer model
        self.model.save(os.path.join(path, 'sentence_transformer'))
        
        # Save the scoring layer
        torch.save({
            'scoring_layer_state_dict': self.scoring_layer.state_dict(),
            'threshold': self.threshold,
            'max_sim': self.max_sim,
            'embedding_dim': self.embedding_dim
        }, os.path.join(path, 'fuzzy_agent.pt'))
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str, device: str = None) -> 'FuzzyAssertionAgent':
        """
        Load a trained model from disk.
        
        Args:
            path: Directory containing the saved model
            device: Device to load the model on
            
        Returns:
            Loaded FuzzyAssertionAgent instance
        """
        import os
        import torch
        
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the sentence transformer model
        model_path = os.path.join(path, 'sentence_transformer')
        model = SentenceTransformer(model_path, device=device)
        
        # Create agent with dummy parameters (will be overridden)
        agent = cls(model_name='dummy', device=device)
        
        # Replace the model
        agent.model = model
        
        # Load the saved state
        checkpoint = torch.load(os.path.join(path, 'fuzzy_agent.pt'), 
                              map_location=device)
        
        # Update agent parameters
        agent.scoring_layer.load_state_dict(checkpoint['scoring_layer_state_dict'])
        agent.threshold = checkpoint['threshold']
        agent.max_sim = checkpoint['max_sim']
        agent.embedding_dim = checkpoint['embedding_dim']
        
        logger.info(f"Model loaded from {path}")
        return agent


def example_usage():
    """Example usage of the FuzzyAssertionAgent."""
    import time
    
    print("Starting FuzzyAssertionAgent example...")
    start_time = time.time()
    
    try:
        # Initialize the agent
        agent = FuzzyAssertionAgent(threshold=0.7)
        
        # Example test cases
        test_cases = [
            # Similar outputs (should validate)
            ("File saved successfully.", "The file was saved successfully."),
            
            # Different but related outputs (might validate depending on threshold)
            ("Error: File not found: data.txt", "File 'data.txt' could not be found."),
            
            # Very different outputs (should not validate)
            ("Operation completed successfully.", "An error occurred during processing."),
            
            # Empty outputs
            ("", ""),
            ("Some output", ""),
        ]
        
        # Run validation
        print("\nRunning validation on test cases:")
        print("=" * 80)
        
        for i, (output_buggy, output_fixed) in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"  Buggy: {output_buggy}")
            print(f"  Fixed: {output_fixed}")
            
            result = agent.validate_output(output_buggy, output_fixed)
            
            print(f"  Valid: {'✅' if result.is_valid else '❌'}")
            print(f"  Confidence: {result.confidence:.4f}")
            print(f"  Semantic Similarity: {result.semantic_similarity:.4f}")
            print(f"  Contextual Similarity: {result.context_similarity:.4f}")
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nExample completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    example_usage()
