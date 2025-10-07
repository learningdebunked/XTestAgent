"""
Implements Equation (1) from Section 3.3.2:
s_vec_i = Encoder_Code(s_i)

Encodes Java methods into vector representations using Transformer models.
"""

import torch
from transformers import AutoTokenizer, AutoModel, BatchEncoding
from typing import List, Dict, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CodeEmbedding:
    """Represents encoded method with metadata"""
    method_name: str
    embedding: np.ndarray
    attention_weights: np.ndarray
    tokens: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'method_name': self.method_name,
            'embedding': self.embedding.tolist(),
            'attention_weights': self.attention_weights.tolist(),
            'tokens': self.tokens
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeEmbedding':
        """Create from dictionary"""
        return cls(
            method_name=data['method_name'],
            embedding=np.array(data['embedding']),
            attention_weights=np.array(data['attention_weights']),
            tokens=data['tokens']
        )

class CodeEncoder:
    """
    Encodes source code into vector representations.
    Implements Equation (1): s_vec = Encoder_Code(s_i)
    """
    
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        """
        Initialize code encoder with pre-trained model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        # Initialize tokenizer and model
        logger.info(f"Loading tokenizer and model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        logger.info(f"Code encoder initialized on {self.device}")
        logger.info(f"Model: {model_name}")
    
    def encode_method(self, method_source: str, 
                     method_name: str = "") -> CodeEmbedding:
        """
        Encode a single method into vector representation.
        
        Implements: s_vec_i = Encoder_Code(s_i) from Equation (1)
        
        Args:
            method_source: Source code of the method
            method_name: Name of the method (for tracking)
            
        Returns:
            CodeEmbedding with vector representation
        """
        try:
            # Tokenize code
            inputs = self.tokenizer(
                method_source,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
                return_attention_mask=True
            ).to(self.device)
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_attentions=True,
                    return_dict=True
                )
            
            # Extract embeddings (CLS token)
            # Shape: (batch_size, hidden_size)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Extract attention weights for explainability
            # Take mean across attention heads
            attention = torch.stack(outputs.attentions[-1:]).mean(dim=0).mean(dim=1).cpu().numpy()
            
            # Get tokens for visualization
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            return CodeEmbedding(
                method_name=method_name,
                embedding=embeddings[0],  # Shape: (hidden_size,)
                attention_weights=attention[0],  # Shape: (seq_len, seq_len)
                tokens=tokens
            )
            
        except Exception as e:
            logger.error(f"Error encoding method {method_name}: {str(e)}")
            raise
    
    def encode_batch(self, methods: List[str], 
                    method_names: Optional[List[str]] = None) -> List[CodeEmbedding]:
        """
        Encode multiple methods in batch for efficiency.
        
        Args:
            methods: List of method source codes
            method_names: Optional list of method names
            
        Returns:
            List of CodeEmbedding objects
        """
        if method_names is None:
            method_names = [f"method_{i}" for i in range(len(methods))]
        
        if len(methods) != len(method_names):
            raise ValueError("methods and method_names must have the same length")
        
        embeddings = []
        
        # Process in batches of 8
        batch_size = 8
        for i in range(0, len(methods), batch_size):
            batch = methods[i:i+batch_size]
            names = method_names[i:i+batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_attention_mask=True
                ).to(self.device)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        output_attentions=True,
                        return_dict=True
                    )
                
                # Process each example in batch
                for j in range(len(batch)):
                    # Extract CLS token embedding
                    emb = outputs.last_hidden_state[j, 0, :].cpu().numpy()
                    
                    # Extract attention (mean across heads)
                    attn = torch.stack(outputs.attentions[-1:]).mean(dim=0)[j].mean(dim=0).cpu().numpy()
                    
                    # Get tokens
                    tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][j])
                    
                    embeddings.append(CodeEmbedding(
                        method_name=names[j],
                        embedding=emb,
                        attention_weights=attn,
                        tokens=tokens
                    ))
                    
            except Exception as e:
                logger.error(f"Error processing batch starting at index {i}: {str(e)}")
                # Fall back to individual processing for this batch
                for code, name in zip(batch, names):
                    try:
                        emb = self.encode_method(code, name)
                        embeddings.append(emb)
                    except Exception as inner_e:
                        logger.error(f"Failed to process {name}: {str(inner_e)}")
        
        return embeddings
    
    def compute_similarity(self, emb1: CodeEmbedding, 
                          emb2: CodeEmbedding) -> float:
        """
        Compute cosine similarity between two code embeddings.
        
        Args:
            emb1, emb2: Code embeddings to compare
            
        Returns:
            Similarity score [0, 1]
        """
        # Cosine similarity
        dot_product = np.dot(emb1.embedding, emb2.embedding)
        norm1 = np.linalg.norm(emb1.embedding)
        norm2 = np.linalg.norm(emb2.embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(max(0.0, min(1.0, (similarity + 1) / 2)))  # Normalize to [0, 1]
    
    def get_attention_heatmap(self, embedding: CodeEmbedding) -> Dict[str, Any]:
        """
        Generate attention heatmap for explainability.
        
        Args:
            embedding: CodeEmbedding instance
            
        Returns:
            Dictionary with tokens and attention scores
        """
        return {
            "tokens": embedding.tokens,
            "attention": embedding.attention_weights.tolist(),
            "method": embedding.method_name
        }
    
    def save_embeddings(self, embeddings: List[CodeEmbedding], 
                       output_path: Union[str, Path]) -> None:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: List of CodeEmbedding objects
            output_path: Path to save the embeddings
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model_name,
            'embeddings': [emb.to_dict() for emb in embeddings]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
    
    @classmethod
    def load_embeddings(cls, file_path: Union[str, Path]) -> List[CodeEmbedding]:
        """
        Load embeddings from disk.
        
        Args:
            file_path: Path to the saved embeddings
            
        Returns:
            List of CodeEmbedding objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return [CodeEmbedding.from_dict(emb) for emb in data['embeddings']]

def example_usage() -> None:
    """Example usage of the CodeEncoder"""
    import time
    
    # Initialize encoder
    encoder = CodeEncoder()
    
    # Sample Java methods
    methods = [
        """
        public int calculateSum(int a, int b) {
            int result = a + b;
            return result;
        }
        """,
        """
        public int multiply(int x, int y) {
            return x * y;
        }
        """
    ]
    
    # Encode methods
    start_time = time.time()
    embeddings = encoder.encode_batch(methods, ["calculateSum", "multiply"])
    
    for emb in embeddings:
        print(f"\nMethod: {emb.method_name}")
        print(f"Embedding shape: {emb.embedding.shape}")
        print(f"First 5 dims: {emb.embedding[:5].round(6)}")
    
    # Compute similarity
    if len(embeddings) >= 2:
        sim = encoder.compute_similarity(embeddings[0], embeddings[1])
        print(f"\nSimilarity between methods: {sim:.4f}")
    
    print(f"\nProcessing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    example_usage()
