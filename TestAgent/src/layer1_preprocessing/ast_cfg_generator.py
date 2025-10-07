"""
Implements Section 3.3.2: Source Code Representation
Generates Abstract Syntax Trees (ASTs) and Control Flow Graphs (CFGs)
for Java source files.
"""

import javalang
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import networkx as nx
from dataclasses import dataclass, asdict
import json

@dataclass
class MethodRepresentation:
    """Represents a single method with its AST and CFG"""
    method_name: str
    class_name: str
    signature: str
    ast: Any  # javalang.tree.MethodDeclaration
    cfg: nx.DiGraph
    source_code: str
    start_line: int
    end_line: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'method_name': self.method_name,
            'class_name': self.class_name,
            'signature': self.signature,
            'source_code': self.source_code,
            'start_line': self.start_line,
            'end_line': self.end_line,
            # AST and CFG need special handling for serialization
        }

class ASTCFGGenerator:
    """
    Generates Abstract Syntax Trees and Control Flow Graphs.
    Corresponds to S = {s1, s2, ..., sn} from paper Section 3.3.2.
    """
    
    def __init__(self):
        self.methods: List[MethodRepresentation] = []
    
    def parse_java_file(self, file_path: Path) -> List[MethodRepresentation]:
        """
        Parse a Java file and extract all methods with AST/CFG.
        
        Args:
            file_path: Path to Java source file
            
        Returns:
            List of MethodRepresentation objects
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        try:
            # Parse Java source to AST
            tree = javalang.parse.parse(source_code)
        except javalang.parser.JavaSyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
            return []
        
        methods: List[MethodRepresentation] = []
        
        # Extract all classes and methods
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            
            for method in node.methods:
                # Generate AST for method
                ast = method
                
                # Generate CFG for method
                cfg = self._build_cfg(method)
                
                # Extract source code for method
                method_source = self._extract_method_source(
                    source_code, method
                )
                
                method_repr = MethodRepresentation(
                    method_name=method.name,
                    class_name=class_name,
                    signature=self._get_method_signature(method),
                    ast=ast,
                    cfg=cfg,
                    source_code=method_source,
                    start_line=method.position.line if method.position else 0,
                    end_line=0  # Will be calculated
                )
                
                methods.append(method_repr)
        
        return methods
    
    def _build_cfg(self, method: javalang.tree.MethodDeclaration) -> nx.DiGraph:
        """
        Build Control Flow Graph for a method.
        
        Returns:
            NetworkX directed graph representing control flow
        """
        cfg = nx.DiGraph()
        
        if not hasattr(method, 'body') or not method.body:
            return cfg
        
        # Start node
        cfg.add_node("START", type="entry")
        current_node = "START"
        node_counter = 0
        
        # Process method body statements
        for stmt in method.body:
            node_counter, current_node = self._process_statement(
                stmt, cfg, current_node, node_counter
            )
        
        # End node
        cfg.add_node("END", type="exit")
        cfg.add_edge(current_node, "END")
        
        return cfg
    
    def _process_statement(self, stmt: Any, cfg: nx.DiGraph, 
                          parent_node: str, counter: int) -> Tuple[int, str]:
        """Process a single statement and add to CFG"""
        node_id = f"node_{counter}"
        counter += 1
        
        if isinstance(stmt, javalang.tree.IfStatement):
            # If statement creates branching
            cfg.add_node(node_id, type="if", condition=str(stmt.condition))
            cfg.add_edge(parent_node, node_id)
            
            # True branch
            true_node = f"node_{counter}"
            counter += 1
            cfg.add_node(true_node, type="then")
            cfg.add_edge(node_id, true_node, label="true")
            
            # False branch
            false_node = f"node_{counter}"
            counter += 1
            cfg.add_node(false_node, type="else")
            cfg.add_edge(node_id, false_node, label="false")
            
            return counter, node_id
        
        elif isinstance(stmt, javalang.tree.WhileStatement):
            # While loop creates cycle
            cfg.add_node(node_id, type="while", condition=str(stmt.condition))
            cfg.add_edge(parent_node, node_id)
            
            # Loop body
            body_node = f"node_{counter}"
            counter += 1
            cfg.add_node(body_node, type="loop_body")
            cfg.add_edge(node_id, body_node, label="true")
            cfg.add_edge(body_node, node_id, label="loop_back")
            
            return counter, node_id
        
        elif isinstance(stmt, javalang.tree.ForStatement):
            # For loop
            cfg.add_node(node_id, type="for")
            cfg.add_edge(parent_node, node_id)
            return counter, node_id
        
        else:
            # Regular statement
            stmt_type = type(stmt).__name__
            cfg.add_node(node_id, type=stmt_type, statement=str(stmt))
            cfg.add_edge(parent_node, node_id)
            return counter, node_id
    
    def _get_method_signature(self, method: javalang.tree.MethodDeclaration) -> str:
        """Generate method signature string"""
        params = ", ".join(
            f"{p.type.name} {p.name}" for p in (method.parameters or [])
        )
        return_type = method.return_type.name if method.return_type else "void"
        return f"{return_type} {method.name}({params})"
    
    def _extract_method_source(self, full_source: str, 
                              method: javalang.tree.MethodDeclaration) -> str:
        """Extract source code for specific method"""
        # Simplified - in production, use proper source mapping
        return f"// Method: {method.name}\n// (source extraction not implemented)"
    
    def analyze_project(self, project_path: Path) -> List[MethodRepresentation]:
        """
        Analyze entire project and extract all methods.
        
        Returns:
            List of all methods with AST/CFG
        """
        all_methods: List[MethodRepresentation] = []
        
        # Find all Java files
        java_files = list(project_path.rglob("*.java"))
        
        for java_file in java_files:
            methods = self.parse_java_file(java_file)
            all_methods.extend(methods)
        
        return all_methods
    
    def export_cfg(self, cfg: nx.DiGraph, output_path: Path) -> None:
        """Export CFG to JSON format for visualization"""
        # Convert CFG to a serializable format
        data = nx.node_link_data(cfg)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

# Example usage
def example_usage() -> None:
    """Example usage of the ASTCFGGenerator"""
    generator = ASTCFGGenerator()
    
    # Analyze a single file
    methods = generator.parse_java_file(Path("./example.java"))
    
    for method in methods:
        print(f"Method: {method.method_name}")
        print(f"Signature: {method.signature}")
        print(f"CFG nodes: {method.cfg.number_of_nodes()}")
        print(f"CFG edges: {method.cfg.number_of_edges()}")
        print("---")

if __name__ == "__main__":
    example_usage()
