"""
Implements Equation (2) from Section 3.3.3:
Delta_sem = Encoder_Code(P_f) - Encoder_Code(P_b)

Analyzes semantic differences between buggy and fixed versions.
"""

from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import difflib
import logging
from dataclasses import dataclass, asdict
import json
import numpy as np

# Import local modules
from .code_encoder import CodeEncoder, CodeEmbedding
from .ast_cfg_generator import ASTCFGGenerator, MethodRepresentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SemanticDiff:
    """Represents semantic difference between versions"""
    modified_methods: List[str]
    semantic_distance: float
    buggy_embedding: CodeEmbedding
    fixed_embedding: CodeEmbedding
    textual_diff: str
    commit_message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'modified_methods': self.modified_methods,
            'semantic_distance': self.semantic_distance,
            'buggy_embedding': self.buggy_embedding.to_dict(),
            'fixed_embedding': self.fixed_embedding.to_dict(),
            'textual_diff': self.textual_diff,
            'commit_message': self.commit_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticDiff':
        """Create from dictionary"""
        return cls(
            modified_methods=data['modified_methods'],
            semantic_distance=data['semantic_distance'],
            buggy_embedding=CodeEmbedding.from_dict(data['buggy_embedding']),
            fixed_embedding=CodeEmbedding.from_dict(data['fixed_embedding']),
            textual_diff=data['textual_diff'],
            commit_message=data['commit_message']
        )

class SemanticDiffAnalyzer:
    """
    Analyzes semantic differences between buggy and fixed code.
    Implements Equation (2): Delta_sem = Encoder(P_f) - Encoder(P_b)
    """
    
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        """
        Initialize the semantic diff analyzer.
        
        Args:
            model_name: Name of the pre-trained model to use for encoding
        """
        self.encoder = CodeEncoder(model_name=model_name)
        self.ast_generator = ASTCFGGenerator()
        logger.info("SemanticDiffAnalyzer initialized")
    
    def analyze_diff(self, buggy_code: str, fixed_code: str,
                    commit_message: str = "") -> SemanticDiff:
        """
        Analyze semantic difference between buggy and fixed versions.
        
        Implements Equation (2) from paper.
        
        Args:
            buggy_code: Source code of buggy version (P_b)
            fixed_code: Source code of fixed version (P_f)
            commit_message: Commit message (M from paper)
            
        Returns:
            SemanticDiff object with analysis
        """
        try:
            logger.info("Analyzing semantic difference between versions")
            
            # Encode both versions
            logger.debug("Encoding buggy version")
            buggy_emb = self.encoder.encode_method(buggy_code, "buggy_version")
            
            logger.debug("Encoding fixed version")
            fixed_emb = self.encoder.encode_method(fixed_code, "fixed_version")
            
            # Compute semantic distance: ||Encoder(P_f) - Encoder(P_b)||
            semantic_distance = self._compute_distance(
                buggy_emb.embedding,
                fixed_emb.embedding
            )
            logger.debug(f"Semantic distance: {semantic_distance:.4f}")
            
            # Generate textual diff
            logger.debug("Generating textual diff")
            textual_diff = self._generate_textual_diff(buggy_code, fixed_code)
            
            # Identify modified methods
            logger.debug("Identifying modified methods")
            modified_methods = self._identify_modified_methods(
                buggy_code, fixed_code
            )
            
            logger.info(f"Analysis complete. Found {len(modified_methods)} modified methods")
            
            return SemanticDiff(
                modified_methods=modified_methods,
                semantic_distance=semantic_distance,
                buggy_embedding=buggy_emb,
                fixed_embedding=fixed_emb,
                textual_diff=textual_diff,
                commit_message=commit_message
            )
            
        except Exception as e:
            logger.error(f"Error analyzing diff: {str(e)}")
            raise
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute L2 distance between embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            L2 distance between the embeddings
        """
        return float(np.linalg.norm(emb1 - emb2))
    
    def _generate_textual_diff(self, code1: str, code2: str) -> str:
        """
        Generate unified diff between two code snippets
        
        Args:
            code1: First code snippet
            code2: Second code snippet
            
        Returns:
            Unified diff string
        """
        diff = difflib.unified_diff(
            code1.splitlines(keepends=True),
            code2.splitlines(keepends=True),
            fromfile='buggy',
            tofile='fixed',
            lineterm=''
        )
        return '\n'.join(diff)
    
    def _identify_modified_methods(self, buggy_code: str, 
                                 fixed_code: str) -> List[str]:
        """
        Identify which methods were modified between versions
        
        Args:
            buggy_code: Source code of buggy version
            fixed_code: Source code of fixed version
            
        Returns:
            List of modified method signatures
        """
        modified = []
        
        # Get method signatures from both versions
        buggy_methods = self._extract_method_signatures(buggy_code)
        fixed_methods = self._extract_method_signatures(fixed_code)
        
        # Find modified methods by comparing signatures and content
        for sig in set(buggy_methods.keys()).union(set(fixed_methods.keys())):
            if sig not in buggy_methods or sig not in fixed_methods:
                modified.append(sig)  # Method added or removed
            elif buggy_methods[sig] != fixed_methods[sig]:
                modified.append(sig)  # Method content changed
                
        return modified
    
    def _extract_method_signatures(self, code: str) -> Dict[str, str]:
        """
        Extract method signatures from Java code
        
        Args:
            code: Java source code
            
        Returns:
            Dictionary mapping method signatures to their content hashes
        """
        try:
            # Use AST parser to get accurate method signatures
            methods = self.ast_generator.parse_java_file(Path("temp.java"), code)
            return {m.signature: str(hash(m.source_code)) for m in methods}
        except Exception as e:
            logger.warning(f"Error parsing methods: {str(e)}")
            # Fallback to simple text-based method extraction
            return self._simple_method_extraction(code)
    
    def _simple_method_extraction(self, code: str) -> Dict[str, str]:
        """
        Simple text-based method extraction as fallback
        """
        methods = {}
        lines = code.split('\n')
        current_method = None
        current_body = []
        brace_count = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Check for method start (simplified)
            if ('public' in stripped or 'private' in stripped or 'protected' in stripped) and 'class' not in stripped and '(' in stripped and ')' in stripped:
                if current_method is not None:
                    methods[current_method] = str(hash('\n'.join(current_body)))
                current_method = stripped.split('{')[0].strip()
                current_body = [stripped]
                brace_count = 1
            elif current_method is not None:
                current_body.append(stripped)
                brace_count += stripped.count('{')
                brace_count -= stripped.count('}')
                if brace_count == 0:
                    methods[current_method] = str(hash('\n'.join(current_body)))
                    current_method = None
        
        return methods
    
    def analyze_project_diff(self, buggy_project: Path, 
                           fixed_project: Path) -> List[SemanticDiff]:
        """
        Analyze differences across entire project.
        
        Args:
            buggy_project: Path to buggy version directory
            fixed_project: Path to fixed version directory
            
        Returns:
            List of SemanticDiff objects for each modified file
        """
        diffs = []
        
        if not buggy_project.exists() or not fixed_project.exists():
            raise ValueError("Both project directories must exist")
        
        logger.info(f"Analyzing project diff between {buggy_project} and {fixed_project}")
        
        # Find all Java files in buggy version
        buggy_files = list(buggy_project.rglob("*.java"))
        logger.info(f"Found {len(buggy_files)} Java files in buggy version")
        
        for buggy_file in buggy_files:
            try:
                # Find corresponding file in fixed version
                rel_path = buggy_file.relative_to(buggy_project)
                fixed_file = fixed_project / rel_path
                
                if not fixed_file.exists():
                    logger.debug(f"Skipping {rel_path}: no corresponding file in fixed version")
                    continue
                
                # Read both versions
                with open(buggy_file, 'r', encoding='utf-8') as f:
                    buggy_code = f.read()
                with open(fixed_file, 'r', encoding='utf-8') as f:
                    fixed_code = f.read()
                
                # Skip if identical
                if buggy_code == fixed_code:
                    continue
                
                logger.info(f"Analyzing changes in {rel_path}")
                
                # Analyze diff
                diff = self.analyze_diff(buggy_code, fixed_code)
                diffs.append(diff)
                
            except Exception as e:
                logger.error(f"Error processing {buggy_file}: {str(e)}")
                continue
        
        logger.info(f"Analysis complete. Found {len(diffs)} modified files")
        return diffs
    
    def get_context_for_generation(self, diff: SemanticDiff) -> Dict[str, Any]:
        """
        Extract context for test generation.
        
        Returns dictionary with:
        - semantic_diff: vector representation
        - modified_methods: list of changed methods
        - commit_message: description of change
        - textual_diff: line-by-line changes
        """
        return {
            "semantic_distance": diff.semantic_distance,
            "modified_methods": diff.modified_methods,
            "commit_message": diff.commit_message,
            "textual_diff": diff.textual_diff,
            "buggy_embedding": diff.buggy_embedding.embedding.tolist(),
            "fixed_embedding": diff.fixed_embedding.embedding.tolist()
        }
    
    def save_diff_analysis(self, diff: SemanticDiff, output_path: Path) -> None:
        """
        Save diff analysis to a file.
        
        Args:
            diff: SemanticDiff object to save
            output_path: Path to save the analysis
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(diff.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved diff analysis to {output_path}")
    
    @classmethod
    def load_diff_analysis(cls, file_path: Path) -> SemanticDiff:
        """
        Load diff analysis from a file.
        
        Args:
            file_path: Path to the saved analysis
            
        Returns:
            Loaded SemanticDiff object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return SemanticDiff.from_dict(data)

def example_usage() -> None:
    """Example usage of the SemanticDiffAnalyzer"""
    import time
    
    print("Starting semantic diff analysis example...")
    start_time = time.time()
    
    # Initialize analyzer
    analyzer = SemanticDiffAnalyzer()
    
    # Example buggy and fixed code
    buggy = """
    public class Calculator {
        public int divide(int a, int b) {
            return a / b;  // Bug: no zero check
        }
        
        public int add(int x, int y) {
            return x + y;
        }
    }
    """
    
    fixed = """
    public class Calculator {
        public int divide(int a, int b) {
            if (b == 0) {
                throw new IllegalArgumentException("Division by zero");
            }
            return a / b;  // Fixed: added zero check
        }
        
        public int add(int x, int y) {
            return x + y;  // Unchanged
        }
    }
    """
    
    # Analyze the diff
    print("\nAnalyzing code changes...")
    diff = analyzer.analyze_diff(
        buggy_code=buggy,
        fixed_code=fixed,
        commit_message="Fix division by zero in Calculator.divide"
    )
    
    # Print results
    print(f"\nAnalysis Results:")
    print(f"- Semantic distance: {diff.semantic_distance:.4f}")
    print(f"- Modified methods: {', '.join(diff.modified_methods) if diff.modified_methods else 'None'}")
    print(f"- Commit message: {diff.commit_message}")
    
    # Save and load example
    output_path = Path("diff_analysis.json")
    print(f"\nSaving analysis to {output_path}...")
    analyzer.save_diff_analysis(diff, output_path)
    
    print(f"\nAnalysis completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    example_usage()
