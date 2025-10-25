"""
Helper methods for parsing source code files in the knowledge graph.

These methods support the main knowledge graph parsers by providing
utilities for extracting specific information from AST nodes.
"""

from typing import Dict, List, Optional, Any
import ast
import javalang
import logging

logger = logging.getLogger(__name__)


def process_java_class(kg_instance, node, file_node: Dict[str, Any], 
                      package_name: str, content: str) -> None:
    """Process a Java class and add it to the knowledge graph.
    
    Args:
        kg_instance: KnowledgeGraph instance
        node: javalang AST node for the class
        file_node: File node in the graph
        package_name: Package name
        content: Full file content
    """
    try:
        class_name = node.name
        
        # Create class node
        class_data = {
            'id': f"class_{package_name}.{class_name}",
            'name': class_name,
            'package': package_name,
            'is_abstract': 'abstract' in (node.modifiers or []),
            'is_interface': isinstance(node, javalang.tree.InterfaceDeclaration),
            'is_enum': isinstance(node, javalang.tree.EnumDeclaration),
            'superclass': node.extends.name if hasattr(node, 'extends') and node.extends else None,
            'interfaces': [i.name for i in (node.implements or [])] if hasattr(node, 'implements') else [],
            'start_line': 0,
            'end_line': 0,
            'docstring': node.documentation or ''
        }
        
        class_graph_node = kg_instance._create_class_node(class_data)
        
        # Link class to file
        kg_instance.constructor.create_relationship(
            class_graph_node['id'],
            file_node['id'],
            kg_instance.constructor.RelationshipType.DEFINED_IN if hasattr(kg_instance.constructor, 'RelationshipType') else 'DEFINED_IN',
            {}
        )
        
        # Process methods
        if hasattr(node, 'methods'):
            for method in node.methods:
                process_java_method(kg_instance, method, class_graph_node, package_name, class_name)
                
    except Exception as e:
        logger.error(f"Error processing Java class: {e}", exc_info=True)


def process_java_method(kg_instance, method, class_node: Dict[str, Any],
                       package_name: str, class_name: str) -> None:
    """Process a Java method and add it to the knowledge graph.
    
    Args:
        kg_instance: KnowledgeGraph instance
        method: javalang method node
        class_node: Class node in the graph
        package_name: Package name
        class_name: Class name
    """
    try:
        method_data = {
            'id': f"method_{package_name}.{class_name}.{method.name}",
            'name': method.name,
            'parameters': [f"{p.type.name} {p.name}" for p in (method.parameters or [])],
            'return_type': method.return_type.name if method.return_type else 'void',
            'visibility': 'public' if 'public' in (method.modifiers or []) else 'private',
            'is_static': 'static' in (method.modifiers or []),
            'is_abstract': 'abstract' in (method.modifiers or []),
            'is_constructor': method.name == class_name,
            'start_line': 0,
            'end_line': 0,
            'docstring': method.documentation or '',
            'source_code': ''
        }
        
        method_node = kg_instance._create_method_node(method_data)
        
        # Link method to class
        from .schema import RelationshipType
        kg_instance.constructor.create_relationship(
            method_node['id'],
            class_node['id'],
            RelationshipType.BELONGS_TO,
            {}
        )
        
    except Exception as e:
        logger.error(f"Error processing Java method: {e}", exc_info=True)


def process_python_class(kg_instance, node: ast.ClassDef, file_node: Dict[str, Any],
                        content: str) -> None:
    """Process a Python class and add it to the knowledge graph.
    
    Args:
        kg_instance: KnowledgeGraph instance
        node: Python AST ClassDef node
        file_node: File node in the graph
        content: Full file content
    """
    try:
        class_data = {
            'id': f"class_{node.name}",
            'name': node.name,
            'package': '',
            'is_abstract': False,
            'is_interface': False,
            'is_enum': False,
            'superclass': node.bases[0].id if node.bases and isinstance(node.bases[0], ast.Name) else None,
            'interfaces': [],
            'start_line': node.lineno,
            'end_line': node.end_lineno or node.lineno,
            'docstring': ast.get_docstring(node) or ''
        }
        
        class_graph_node = kg_instance._create_class_node(class_data)
        
        # Link class to file
        from .schema import RelationshipType
        kg_instance.constructor.create_relationship(
            class_graph_node['id'],
            file_node['id'],
            RelationshipType.DEFINED_IN,
            {}
        )
        
        # Process methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                process_python_method(kg_instance, item, class_graph_node, node.name)
                
    except Exception as e:
        logger.error(f"Error processing Python class: {e}", exc_info=True)


def process_python_method(kg_instance, method: ast.FunctionDef, 
                         class_node: Dict[str, Any], class_name: str) -> None:
    """Process a Python method and add it to the knowledge graph.
    
    Args:
        kg_instance: KnowledgeGraph instance
        method: Python AST FunctionDef node
        class_node: Class node in the graph
        class_name: Class name
    """
    try:
        # Extract parameters
        params = []
        for arg in method.args.args:
            params.append(arg.arg)
        
        method_data = {
            'id': f"method_{class_name}.{method.name}",
            'name': method.name,
            'parameters': params,
            'return_type': 'Any',  # Python doesn't require type hints
            'visibility': 'public' if not method.name.startswith('_') else 'private',
            'is_static': any(isinstance(d, ast.Name) and d.id == 'staticmethod' 
                           for d in method.decorator_list),
            'is_abstract': any(isinstance(d, ast.Name) and d.id == 'abstractmethod' 
                             for d in method.decorator_list),
            'is_constructor': method.name == '__init__',
            'start_line': method.lineno,
            'end_line': method.end_lineno or method.lineno,
            'docstring': ast.get_docstring(method) or '',
            'source_code': ''
        }
        
        method_node = kg_instance._create_method_node(method_data)
        
        # Link method to class
        from .schema import RelationshipType
        kg_instance.constructor.create_relationship(
            method_node['id'],
            class_node['id'],
            RelationshipType.BELONGS_TO,
            {}
        )
        
    except Exception as e:
        logger.error(f"Error processing Python method: {e}", exc_info=True)


def process_python_function(kg_instance, func: ast.FunctionDef, 
                           file_node: Dict[str, Any], content: str) -> None:
    """Process a top-level Python function.
    
    Args:
        kg_instance: KnowledgeGraph instance
        func: Python AST FunctionDef node
        file_node: File node in the graph
        content: Full file content
    """
    try:
        # Extract parameters
        params = []
        for arg in func.args.args:
            params.append(arg.arg)
        
        method_data = {
            'id': f"function_{func.name}",
            'name': func.name,
            'parameters': params,
            'return_type': 'Any',
            'visibility': 'public',
            'is_static': True,
            'is_abstract': False,
            'is_constructor': False,
            'start_line': func.lineno,
            'end_line': func.end_lineno or func.lineno,
            'docstring': ast.get_docstring(func) or '',
            'source_code': ''
        }
        
        method_node = kg_instance._create_method_node(method_data)
        
        # Link function to file
        from .schema import RelationshipType
        kg_instance.constructor.create_relationship(
            method_node['id'],
            file_node['id'],
            RelationshipType.DEFINED_IN,
            {}
        )
        
    except Exception as e:
        logger.error(f"Error processing Python function: {e}", exc_info=True)


def create_python_test_node(kg_instance, func: ast.FunctionDef, 
                           class_node: Optional[Dict[str, Any]], 
                           file_node: Dict[str, Any]) -> None:
    """Create a test node for a Python test function/method.
    
    Args:
        kg_instance: KnowledgeGraph instance
        func: Python AST FunctionDef node
        class_node: Class node if method, None if function
        file_node: File node in the graph
    """
    try:
        # Determine framework
        framework = 'pytest'
        for decorator in func.decorator_list:
            if isinstance(decorator, ast.Name):
                if 'unittest' in decorator.id:
                    framework = 'unittest'
        
        # Extract parameters
        params = [arg.arg for arg in func.args.args if arg.arg != 'self']
        
        test_data = {
            'id': f"test_{func.name}",
            'name': func.name,
            'file_path': file_node.get('path', ''),
            'framework': framework,
            'is_parameterized': any(isinstance(d, ast.Name) and 'parametrize' in d.id 
                                   for d in func.decorator_list),
            'parameters': params,
            'start_line': func.lineno,
            'end_line': func.end_lineno or func.lineno,
            'docstring': ast.get_docstring(func) or ''
        }
        
        test_node = kg_instance._create_test_node(test_data)
        
        # Link test to class or file
        from .schema import RelationshipType
        if class_node:
            kg_instance.constructor.create_relationship(
                test_node['id'],
                class_node['id'],
                RelationshipType.BELONGS_TO,
                {}
            )
        else:
            kg_instance.constructor.create_relationship(
                test_node['id'],
                file_node['id'],
                RelationshipType.DEFINED_IN,
                {}
            )
            
    except Exception as e:
        logger.error(f"Error creating Python test node: {e}", exc_info=True)
