"""
Data Importer for Knowledge Graph.

Handles importing code, tests, and other data into the knowledge graph.
"""

import ast
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import hashlib
import javalang
import astor

from .schema import (
    NodeType, RelationshipType, MethodNode, ClassNode, 
    TestNode, FileNode, PackageNode, TestCaseNode, NodeProperties
)

class DataImporter:
    """Handles importing various data sources into the knowledge graph."""
    
    def __init__(self, graph_constructor):
        """Initialize the data importer.
        
        Args:
            graph_constructor: Instance of GraphConstructor
        """
        self.graph = graph_constructor
        self.logger = logging.getLogger(f"{__name__}.DataImporter")
        
        # Language-specific importers
        self._importers = {
            '.java': self._import_java_file,
            '.py': self._import_python_file,
            '.js': self._import_javascript_file,
            '.ts': self._import_typescript_file,
        }
    
    def import_codebase(self, root_path: str) -> Dict[str, int]:
        """Import an entire codebase into the knowledge graph.
        
        Args:
            root_path: Path to the root of the codebase
            
        Returns:
            Dictionary with import statistics
        """
        stats = {
            'files_processed': 0,
            'methods_imported': 0,
            'classes_imported': 0,
            'tests_imported': 0,
            'errors': 0
        }
        
        root_path = Path(root_path).resolve()
        
        # Walk through the directory tree
        for file_path in root_path.rglob('*'):
            if not file_path.is_file():
                continue
                
            try:
                # Skip binary files and other non-code files
                if self._should_skip_file(file_path):
                    continue
                
                # Get the appropriate importer based on file extension
                importer = self._importers.get(file_path.suffix.lower())
                if not importer:
                    self.logger.debug(f"No importer for file: {file_path}")
                    continue
                
                # Import the file
                result = importer(file_path, root_path)
                
                # Update statistics
                stats['files_processed'] += 1
                stats['methods_imported'] += result.get('methods', 0)
                stats['classes_imported'] += result.get('classes', 0)
                stats['tests_imported'] += result.get('tests', 0)
                
            except Exception as e:
                self.logger.error(f"Error importing {file_path}: {e}", exc_info=True)
                stats['errors'] += 1
        
        return stats
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Determine if a file should be skipped during import."""
        # Skip hidden files and directories
        if any(part.startswith('.') for part in file_path.parts):
            return True
            
        # Skip common non-code directories
        skip_dirs = {'__pycache__', 'node_modules', 'target', 'build', 'dist', '.git'}
        if any(part in skip_dirs for part in file_path.parts):
            return True
            
        # Check file extensions
        valid_extensions = {'.java', '.py', '.js', '.ts', '.jsx', '.tsx'}
        if file_path.suffix.lower() not in valid_extensions:
            return True
            
        # Check file size (skip large files)
        max_size = 10 * 1024 * 1024  # 10 MB
        if file_path.stat().st_size > max_size:
            self.logger.warning(f"Skipping large file: {file_path}")
            return True
            
        return False
    
    def _import_java_file(self, file_path: Path, root_path: Path) -> Dict[str, int]:
        """Import a Java source file into the knowledge graph."""
        stats = {'methods': 0, 'classes': 0, 'tests': 0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse the Java file
            tree = javalang.parse.parse(source_code)
            
            # Create file node
            rel_path = file_path.relative_to(root_path)
            file_node = FileNode(
                path=str(rel_path),
                file_name=file_path.name,
                extension=file_path.suffix,
                language='java',
                size_bytes=file_path.stat().st_size,
                last_modified=file_path.stat().st_mtime,
                checksum=self._calculate_checksum(file_path)
            )
            
            # Add file to graph
            file_id = self.graph.add_node(NodeType.FILE, file_node.__dict__)
            
            # Process each class in the file
            for path, node in tree.filter(javalang.tree.ClassDeclaration):
                is_test = any(m.annotation.name.lower() == 'test' 
                            for m in (node.annotations or []))
                
                # Create class node
                class_node = ClassNode(
                    name=node.name,
                    package=tree.package.name if hasattr(tree, 'package') else '',
                    is_interface=isinstance(node, javalang.tree.InterfaceDeclaration),
                    is_abstract='abstract' in node.modifiers,
                    file_path=str(rel_path),
                    documentation=self._extract_java_docs(node.documentation)
                )
                
                class_id = self.graph.add_node(
                    NodeType.CLASS, 
                    class_node.__dict__,
                    is_test=is_test
                )
                
                # Add relationship: File -> Class
                self.graph.add_relationship(
                    file_id, 
                    class_id, 
                    RelationshipType.CONTAINS
                )
                
                stats['classes'] += 1
                
                # Process methods in the class
                for member in node.body:
                    if isinstance(member, javalang.tree.MethodDeclaration):
                        method_node = self._create_java_method_node(
                            member, 
                            class_node.package, 
                            str(rel_path),
                            is_test
                        )
                        
                        method_id = self.graph.add_node(
                            NodeType.METHOD, 
                            method_node.__dict__,
                            is_test=is_test
                        )
                        
                        # Add relationship: Class -> Method
                        self.graph.add_relationship(
                            class_id, 
                            method_id, 
                            RelationshipType.CONTAINS
                        )
                        
                        stats['methods'] += 1
                        
                        # If it's a test method, add test node
                        if is_test or self._is_test_method(member):
                            test_node = TestNode(
                                name=member.name,
                                class_name=class_node.name,
                                package=class_node.package,
                                file_path=str(rel_path),
                                test_framework='junit',
                                is_parameterized=self._is_parameterized_test(member)
                            )
                            
                            test_id = self.graph.add_node(
                                NodeType.TEST, 
                                test_node.__dict__
                            )
                            
                            # Add relationship: Test -> Method
                            self.graph.add_relationship(
                                test_id, 
                                method_id, 
                                RelationshipType.TESTS
                            )
                            
                            stats['tests'] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error parsing Java file {file_path}: {e}", exc_info=True)
            return stats
    
    def _import_python_file(self, file_path: Path, root_path: Path) -> Dict[str, int]:
        """Import a Python source file into the knowledge graph."""
        stats = {'methods': 0, 'classes': 0, 'tests': 0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
                
            # Parse the Python file
            tree = ast.parse(source_code, str(file_path))
            
            # Create file node
            rel_path = file_path.relative_to(root_path)
            file_node = FileNode(
                path=str(rel_path),
                file_name=file_path.name,
                extension=file_path.suffix,
                language='python',
                size_bytes=file_path.stat().st_size,
                last_modified=file_path.stat().st_mtime,
                checksum=self._calculate_checksum(file_path)
            )
            
            # Add file to graph
            file_id = self.graph.add_node(NodeType.FILE, file_node.__dict__)
            
            # Get the module name (package)
            module_parts = list(rel_path.parts[:-1])
            if rel_path.stem != '__init__':
                module_parts.append(rel_path.stem)
            module_name = '.'.join(module_parts)
            
            # Process each class in the file
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    is_test = (
                        node.name.startswith('Test') or 
                        node.name.endswith('Test') or
                        any(d.id == 'unittest.TestCase' 
                            for d in node.bases 
                            if isinstance(d, ast.Name))
                    )
                    
                    # Create class node
                    class_node = ClassNode(
                        name=node.name,
                        package=module_name,
                        is_interface=False,  # Python doesn't have interfaces like Java
                        is_abstract=any(
                            d.id == 'ABC' or 
                            (isinstance(d, ast.Attribute) and d.attr == 'ABC')
                            for d in node.decorator_list
                        ),
                        file_path=str(rel_path),
                        documentation=ast.get_docstring(node)
                    )
                    
                    class_id = self.graph.add_node(
                        NodeType.CLASS, 
                        class_node.__dict__,
                        is_test=is_test
                    )
                    
                    # Add relationship: File -> Class
                    self.graph.add_relationship(
                        file_id, 
                        class_id, 
                        RelationshipType.CONTAINS
                    )
                    
                    stats['classes'] += 1
                    
                    # Process methods in the class
                    for member in node.body:
                        if isinstance(member, ast.FunctionDef):
                            method_node = self._create_python_method_node(
                                member, 
                                module_name, 
                                str(rel_path),
                                is_test
                            )
                            
                            method_id = self.graph.add_node(
                                NodeType.METHOD, 
                                method_node.__dict__,
                                is_test=is_test
                            )
                            
                            # Add relationship: Class -> Method
                            self.graph.add_relationship(
                                class_id, 
                                method_id, 
                                RelationshipType.CONTAINS
                            )
                            
                            stats['methods'] += 1
                            
                            # If it's a test method, add test node
                            is_test_method = is_test and member.name.startswith('test_')
                            if is_test_method:
                                test_node = TestNode(
                                    name=member.name,
                                    class_name=class_node.name,
                                    package=class_node.package,
                                    file_path=str(rel_path),
                                    test_framework='pytest',
                                    is_parameterized='parameterized' in [
                                        d.func.attr if isinstance(d, ast.Call) and hasattr(d.func, 'attr') else ''
                                        for d in member.decorator_list
                                    ]
                                )
                                
                                test_id = self.graph.add_node(
                                    NodeType.TEST, 
                                    test_node.__dict__
                                )
                                
                                # Add relationship: Test -> Method
                                self.graph.add_relationship(
                                    test_id, 
                                    method_id, 
                                    RelationshipType.TESTS
                                )
                                
                                stats['tests'] += 1
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error parsing Python file {file_path}: {e}", exc_info=True)
            return stats
    
    def _import_javascript_file(self, file_path: Path, root_path: Path) -> Dict[str, int]:
        """Import a JavaScript/TypeScript file into the knowledge graph."""
        # This is a simplified implementation
        # In a real-world scenario, you would use a proper JavaScript parser
        # like esprima or @babel/parser
        
        stats = {'methods': 0, 'classes': 0, 'tests': 0}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Create file node
            rel_path = file_path.relative_to(root_path)
            file_node = FileNode(
                path=str(rel_path),
                file_name=file_path.name,
                extension=file_path.suffix,
                language='typescript' if file_path.suffix == '.ts' else 'javascript',
                size_bytes=file_path.stat().st_size,
                last_modified=file_path.stat().st_mtime,
                checksum=self._calculate_checksum(file_path)
            )
            
            # Add file to graph
            file_id = self.graph.add_node(NodeType.FILE, file_node.__dict__)
            
            # In a real implementation, you would parse the JavaScript/TypeScript
            # code to extract classes, methods, and tests
            # This is just a placeholder
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error parsing JavaScript file {file_path}: {e}", exc_info=True)
            return stats
    
    def _import_typescript_file(self, file_path: Path, root_path: Path) -> Dict[str, int]:
        """Import a TypeScript file into the knowledge graph."""
        # TypeScript files are processed the same way as JavaScript files
        # but with type information
        return self._import_javascript_file(file_path, root_path)
    
    # Helper methods for Java import
    def _create_java_method_node(
        self, 
        method: javalang.tree.MethodDeclaration, 
        package: str,
        file_path: str,
        is_test: bool = False
    ) -> MethodNode:
        """Create a MethodNode from a Java method declaration."""
        return MethodNode(
            name=method.name,
            signature=f"{method.return_type} {method.name}({', '.join(f'{p.type} {p.name}' for p in method.parameters)})",
            return_type=str(method.return_type) if method.return_type else 'void',
            parameters=[
                {'name': p.name, 'type': str(p.type)}
                for p in method.parameters
            ],
            visibility=method.modifiers[0] if method.modifiers else 'package',
            is_static='static' in method.modifiers,
            is_abstract='abstract' in method.modifiers,
            is_constructor=isinstance(method, javalang.tree.ConstructorDeclaration),
            start_line=method.position.line if hasattr(method, 'position') and method.position else None,
            end_line=method.position.line if hasattr(method, 'position') and method.position else None,
            file_path=file_path,
            documentation=self._extract_java_docs(method.documentation)
        )
    
    def _is_test_method(self, method: javalang.tree.MethodDeclaration) -> bool:
        """Check if a Java method is a test method."""
        return any(
            annotation.name.lower() == 'test' 
            for annotation in method.annotations or []
        )
    
    def _is_parameterized_test(self, method: javalang.tree.MethodDeclaration) -> bool:
        """Check if a Java test method is parameterized."""
        return any(
            annotation.name.lower() in ['parameterizedtest', 'testwith']
            for annotation in method.annotations or []
        )
    
    def _extract_java_docs(self, docs: Optional[str]) -> Optional[str]:
        """Extract documentation from JavaDoc comments."""
        if not docs:
            return None
            
        # Simple cleanup of JavaDoc comments
        lines = []
        for line in docs.split('\n'):
            line = line.strip()
            if line.startswith('*'):
                line = line[1:].strip()
            if line.startswith('/**') or line.endswith('*/'):
                continue
            if line.startswith('@'):
                break  # Skip tags for now
            lines.append(line)
            
        return '\n'.join(lines).strip() or None
    
    # Helper methods for Python import
    def _create_python_method_node(
        self, 
        method: ast.FunctionDef, 
        module_name: str,
        file_path: str,
        is_test: bool = False
    ) -> MethodNode:
        """Create a MethodNode from a Python function definition."""
        # Get the function signature
        args = []
        for arg in method.args.args:
            arg_type = 'Any'
            if arg.annotation:
                arg_type = astor.to_source(arg.annotation).strip()
            args.append({'name': arg.arg, 'type': arg_type})
        
        # Get return type
        return_type = 'Any'
        if method.returns:
            return_type = astor.to_source(method.returns).strip()
        
        # Get decorators
        decorators = []
        for decorator in method.decorator_list:
            decorator_src = astor.to_source(decorator).strip()
            decorators.append(decorator_src)
        
        return MethodNode(
            name=method.name,
            signature=f"def {method.name}({', '.join(f"{a['name']}: {a['type']}" for a in args)}) -> {return_type}",
            return_type=return_type,
            parameters=args,
            visibility='public',  # Python doesn't have visibility modifiers
            is_static='@staticmethod' in decorators or '@classmethod' in decorators,
            is_abstract='@abstractmethod' in decorators,
            is_constructor=method.name == '__init__',
            start_line=method.lineno,
            end_line=method.end_lineno if hasattr(method, 'end_lineno') else None,
            file_path=file_path,
            documentation=ast.get_docstring(method),
            source_code=astor.to_source(method) if is_test else None
        )
    
    # Utility methods
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate a checksum for a file."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()


# Example usage
if __name__ == "__main__":
    import logging
    from graph_constructor import GraphConstructor
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize the graph constructor
    constructor = GraphConstructor(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password",
        database="testagentx"
    )
    
    # Initialize the data importer
    importer = DataImporter(constructor)
    
    # Import a codebase
    stats = importer.import_codebase("/path/to/your/codebase")
    print(f"Import completed: {stats}")
