"""
Unit tests for the DataImporter class and its implementations.

These tests verify the functionality of importing code from different programming languages
into the knowledge graph.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.layer5_knowledge_graph.data_importer import (
    DataImporter, 
    JavaImporter, 
    PythonImporter,
    JavaScriptImporter
)
from src.layer5_knowledge_graph.schema import NodeType, RelationshipType

class TestDataImporter:
    """Test cases for the base DataImporter class."""
    
    def test_base_class_abstract(self):
        """Test that DataImporter is an abstract base class."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DataImporter(None)
    
    def test_get_importer_java(self):
        """Test getting a Java importer."""
        kg_mock = MagicMock()
        importer = DataImporter.get_importer("java", kg_mock)
        assert isinstance(importer, JavaImporter)
        assert importer.kg == kg_mock
    
    def test_get_importer_python(self):
        """Test getting a Python importer."""
        kg_mock = MagicMock()
        importer = DataImporter.get_importer("python", kg_mock)
        assert isinstance(importer, PythonImporter)
        assert importer.kg == kg_mock
    
    def test_get_importer_javascript(self):
        """Test getting a JavaScript importer."""
        kg_mock = MagicMock()
        importer = DataImporter.get_importer("javascript", kg_mock)
        assert isinstance(importer, JavaScriptImporter)
        assert importer.kg == kg_mock
    
    def test_get_importer_unsupported_language(self):
        """Test getting an importer for an unsupported language."""
        with pytest.raises(ValueError, match="Unsupported language: ruby"):
            DataImporter.get_importer("ruby", MagicMock())

class TestJavaImporter:
    """Test cases for the JavaImporter class."""
    
    @pytest.fixture
    def java_importer(self):
        """Create a JavaImporter with a mock knowledge graph."""
        kg_mock = MagicMock()
        return JavaImporter(kg_mock)
    
    def test_import_codebase(self, java_importer, tmp_path):
        """Test importing a Java codebase."""
        # Create a simple Java project structure
        src_dir = tmp_path / "src" / "main" / "java" / "com" / "example"
        src_dir.mkdir(parents=True)
        
        # Create a simple Java file
        java_file = src_dir / "Calculator.java"
        java_file.write_text(
            """
            package com.example;
            
            /**
             * A simple calculator class.
             */
            public class Calculator {
                /**
                 * Adds two numbers.
                 * @param a First number
                 * @param b Second number
                 * @return Sum of a and b
                 */
                public int add(int a, int b) {
                    return a + b;
                }
            }
            """
        )
        
        # Create a test file
        test_dir = tmp_path / "src" / "test" / "java" / "com" / "example"
        test_dir.mkdir(parents=True)
        
        test_file = test_dir / "CalculatorTest.java"
        test_file.write_text(
            """
            package com.example;
            
            import org.junit.Test;
            import static org.junit.Assert.*;
            
            public class CalculatorTest {
                @Test
                public void testAdd() {
                    Calculator calc = new Calculator();
                    assertEquals(5, calc.add(2, 3));
                }
            }
            """
        )
        
        # Import the codebase
        with patch.object(java_importer, '_process_java_file') as mock_process:
            java_importer.import_codebase(tmp_path)
            
            # Verify that _process_java_file was called for both files
            assert mock_process.call_count == 2
            
            # Verify the file paths
            processed_files = [call.args[0] for call in mock_process.call_args_list]
            assert str(java_file) in processed_files
            assert str(test_file) in processed_files
    
    def test_process_java_file(self, java_importer, tmp_path):
        """Test processing a single Java file."""
        # Create a simple Java file
        java_file = tmp_path / "Calculator.java"
        java_file.write_text(
            """
            package com.example;
            
            /**
             * A simple calculator class.
             */
            public class Calculator {
                /**
                 * Adds two numbers.
                 * @param a First number
                 * @param b Second number
                 * @return Sum of a and b
                 */
                public int add(int a, int b) {
                    return a + b;
                }
            }
            """
        )
        
        # Mock the javalang parser
        with patch('javalang.parse.parse') as mock_parse:
            # Create a mock AST
            mock_ast = MagicMock()
            mock_ast.package.name = 'com.example'
            
            # Mock the class definition
            mock_class = MagicMock()
            mock_class.name = 'Calculator'
            mock_class.documentation = '/**\n             * A simple calculator class.\n             */'
            
            # Mock the method definition
            mock_method = MagicMock()
            mock_method.name = 'add'
            mock_method.documentation = '/**\n                 * Adds two numbers.\n                 * @param a First number\n                 * @param b Second number\n                 * @return Sum of a and b\n                 */'
            mock_method.return_type = 'int'
            mock_method.parameters = [
                MagicMock(type='int', name='a'),
                MagicMock(type='int', name='b')
            ]
            
            mock_class.methods = [mock_method]
            mock_ast.types = [mock_class]
            mock_parse.return_value = mock_ast
            
            # Process the file
            java_importer._process_java_file(java_file)
            
            # Verify that nodes were created for the class and method
            assert java_importer.kg.create_node.call_count >= 2
            
            # Verify the class node was created
            class_calls = [c[1] for c in java_importer.kg.create_node.call_args_list 
                          if c[1]['node_type'] == NodeType.CLASS]
            assert len(class_calls) > 0
            assert class_calls[0]['properties']['name'] == 'Calculator'
            assert 'simple calculator' in class_calls[0]['properties']['docstring']
            
            # Verify the method node was created
            method_calls = [c[1] for c in java_importer.kg.create_node.call_args_list 
                           if c[1]['node_type'] == NodeType.METHOD]
            assert len(method_calls) > 0
            assert method_calls[0]['properties']['name'] == 'add'
            assert 'Adds two numbers' in method_calls[0]['properties']['docstring']
            assert method_calls[0]['properties']['return_type'] == 'int'

class TestPythonImporter:
    """Test cases for the PythonImporter class."""
    
    @pytest.fixture
    def python_importer(self):
        """Create a PythonImporter with a mock knowledge graph."""
        kg_mock = MagicMock()
        return PythonImporter(kg_mock)
    
    def test_import_codebase(self, python_importer, tmp_path):
        """Test importing a Python codebase."""
        # Create a simple Python package
        pkg_dir = tmp_path / "mypackage"
        pkg_dir.mkdir()
        
        # Create __init__.py
        (pkg_dir / "__init__.py").write_text(""""A simple Python package.""")
        
        # Create a module
        (pkg_dir / "calculator.py").write_text(
            """
            """A simple calculator module."""
            
            def add(a: int, b: int) -> int:
                """Add two numbers.
                
                Args:
                    a: First number
                    b: Second number
                    
                Returns:
                    Sum of a and b
                """
                return a + b
            
            class Calculator:
                """A simple calculator class."""
                
                def multiply(self, a: float, b: float) -> float:
                    """Multiply two numbers."""
                    return a * b
            """
        )
        
        # Create a test file
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        
        (tests_dir / "test_calculator.py").write_text(
            """
            import unittest
            from mypackage.calculator import add, Calculator
            
            class TestCalculator(unittest.TestCase):
                def test_add(self):
                    """Test the add function."""
                    self.assertEqual(add(2, 3), 5)
                
                def test_multiply(self):
                    """Test the multiply method."""
                    calc = Calculator()
                    self.assertEqual(calc.multiply(2, 3), 6)
            """
        )
        
        # Import the codebase
        with patch.object(python_importer, '_process_python_file') as mock_process:
            python_importer.import_codebase(tmp_path)
            
            # Verify that _process_python_file was called for all Python files
            assert mock_process.call_count == 3  # __init__.py, calculator.py, test_calculator.py
    
    def test_process_python_file(self, python_importer, tmp_path):
        """Test processing a single Python file."""
        # Create a simple Python file
        py_file = tmp_path / "calculator.py"
        py_file.write_text(
            """
            """A simple calculator module."""
            
            def add(a: int, b: int) -> int:
                """Add two numbers.
                
                Args:
                    a: First number
                    b: Second number
                    
                Returns:
                    Sum of a and b
                """
                return a + b
            
            class Calculator:
                """A simple calculator class."""
                
                def multiply(self, a: float, b: float) -> float:
                    """Multiply two numbers."""
                    return a * b
            """
        )
        
        # Mock the AST parsing
        with patch('ast.parse') as mock_parse, \
             patch('inspect.getsource') as mock_getsource:
            
            # Set up the mock AST
            mock_ast_module = MagicMock()
            mock_ast_module.body = []
            
            # Mock the function definition
            mock_func_def = MagicMock()
            mock_func_def.name = 'add'
            mock_func_def.lineno = 3
            mock_func_def.end_lineno = 14
            mock_func_def.body = [MagicMock()]  # Dummy body
            
            # Mock the class definition
            mock_class_def = MagicMock()
            mock_class_def.name = 'Calculator'
            mock_class_def.lineno = 16
            mock_class_def.end_lineno = 22
            mock_class_def.body = [MagicMock()]  # Dummy body
            
            mock_ast_module.body = [mock_func_def, mock_class_def]
            mock_parse.return_value = mock_ast_module
            
            # Mock the source code
            mock_getsource.return_value = py_file.read_text()
            
            # Process the file
            python_importer._process_python_file(py_file)
            
            # Verify that nodes were created for the function and class
            assert python_importer.kg.create_node.call_count >= 2
            
            # Verify the function node was created
            func_calls = [c[1] for c in python_importer.kg.create_node.call_args_list 
                         if c[1]['node_type'] == NodeType.FUNCTION]
            assert len(func_calls) > 0
            assert func_calls[0]['properties']['name'] == 'add'
            assert 'Add two numbers' in func_calls[0]['properties']['docstring']
            
            # Verify the class node was created
            class_calls = [c[1] for c in python_importer.kg.create_node.call_args_list 
                          if c[1]['node_type'] == NodeType.CLASS]
            assert len(class_calls) > 0
            assert class_calls[0]['properties']['name'] == 'Calculator'
            assert 'simple calculator' in class_calls[0]['properties']['docstring']

class TestJavaScriptImporter:
    """Test cases for the JavaScriptImporter class."""
    
    @pytest.fixture
    def js_importer(self):
        """Create a JavaScriptImporter with a mock knowledge graph."""
        kg_mock = MagicMock()
        return JavaScriptImporter(kg_mock)
    
    def test_import_codebase(self, js_importer, tmp_path):
        """Test importing a JavaScript codebase."""
        # Create a simple JavaScript project
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        
        # Create a simple JavaScript file
        (src_dir / "calculator.js").write_text(
            """
            /**
             * A simple calculator module.
             * @module calculator
             */
            
            /**
             * Adds two numbers.
             * @param {number} a - The first number.
             * @param {number} b - The second number.
             * @returns {number} The sum of a and b.
             */
            function add(a, b) {
                return a + b;
            }
            
            /**
             * A simple calculator class.
             */
            class Calculator {
                /**
                 * Multiplies two numbers.
                 * @param {number} a - The first number.
                 * @param {number} b - The second number.
                 * @returns {number} The product of a and b.
                 */
                multiply(a, b) {
                    return a * b;
                }
            }
            
            module.exports = { add, Calculator };
            """
        )
        
        # Create a test file
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        
        (test_dir / "calculator.test.js").write_text(
            """
            const { add, Calculator } = require('../src/calculator');
            
            describe('Calculator', () => {
                test('adds 2 + 3 to equal 5', () => {
                    expect(add(2, 3)).toBe(5);
                });
                
                test('multiplies 2 * 3 to equal 6', () => {
                    const calc = new Calculator();
                    expect(calc.multiply(2, 3)).toBe(6);
                });
            });
            """
        )
        
        # Import the codebase
        with patch.object(js_importer, '_process_javascript_file') as mock_process:
            js_importer.import_codebase(tmp_path)
            
            # Verify that _process_javascript_file was called for both files
            assert mock_process.call_count == 2
            
            # Verify the file paths
            processed_files = [call.args[0] for call in mock_process.call_args_list]
            assert str(src_dir / "calculator.js") in processed_files
            assert str(test_dir / "calculator.test.js") in processed_files
    
    def test_process_javascript_file(self, js_importer, tmp_path):
        """Test processing a single JavaScript file."""
        # Create a simple JavaScript file
        js_file = tmp_path / "calculator.js"
        js_file.write_text(
            """
            /**
             * A simple calculator module.
             * @module calculator
             */
            
            /**
             * Adds two numbers.
             * @param {number} a - The first number.
             * @param {number} b - The second number.
             * @returns {number} The sum of a and b.
             */
            function add(a, b) {
                return a + b;
            }
            
            /**
             * A simple calculator class.
             */
            class Calculator {
                /**
                 * Multiplies two numbers.
                 * @param {number} a - The first number.
                 * @param {number} b - The second number.
                 * @returns {number} The product of a and b.
                 */
                multiply(a, b) {
                    return a * b;
                }
            }
            
            module.exports = { add, Calculator };
            """
        )
        
        # Process the file
        with patch('esprima.parseScript') as mock_parse:
            # Mock the AST
            mock_ast = MagicMock()
            mock_parse.return_value = mock_ast
            
            # Process the file
            js_importer._process_javascript_file(js_file)
            
            # Verify that nodes were created for the function and class
            assert js_importer.kg.create_node.call_count >= 2
            
            # Verify the function node was created
            func_calls = [c[1] for c in js_importer.kg.create_node.call_args_list 
                         if c[1]['node_type'] == NodeType.FUNCTION]
            assert len(func_calls) > 0
            assert func_calls[0]['properties']['name'] == 'add'
            
            # Verify the class node was created
            class_calls = [c[1] for c in js_importer.kg.create_node.call_args_list 
                          if c[1]['node_type'] == NodeType.CLASS]
            assert len(class_calls) > 0
            assert class_calls[0]['properties']['name'] == 'Calculator'
