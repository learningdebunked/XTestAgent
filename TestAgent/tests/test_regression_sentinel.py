"""Unit tests for the Regression Sentinel module."""

import unittest
import tempfile
import os
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

from src.layer4_patch_regression.regression_sentinel_agent import (
    RegressionSentinelAgent,
    TestRelevancePrediction,
    GATModel
)

# Mock TestCase class
class MockTestCase:
    def __init__(self, test_id, file_path):
        self.test_id = test_id
        self.file_path = file_path

class TestRegressionSentinel(unittest.TestCase):
    """Test cases for the Regression Sentinel Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'device': 'cpu',
            'threshold': 0.7,
            'min_confidence': 0.6
        }
        self.agent = RegressionSentinelAgent(self.config)
        
        # Create a simple project structure
        self.project_path = os.path.join(self.temp_dir, 'test_project')
        os.makedirs(os.path.join(self.project_path, 'src', 'main', 'java', 'com', 'example'), exist_ok=True)
        os.makedirs(os.path.join(self.project_path, 'src', 'test', 'java', 'com', 'example'), exist_ok=True)
        
        # Create a simple Java class
        with open(os.path.join(self.project_path, 'src', 'main', 'java', 'com', 'example', 'Calculator.java'), 'w') as f:
            f.write("""
            package com.example;
            
            public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }
                
                public int subtract(int a, int b) {
                    return a - b;
                }
            }
            """)
        
        # Create a simple test class
        with open(os.path.join(self.project_path, 'src', 'test', 'java', 'com', 'example', 'CalculatorTest.java'), 'w') as f:
            f.write("""
            package com.example;
            
            import org.junit.Test;
            import static org.junit.Assert.*;
            
            public class CalculatorTest {
                @Test
                public void testAdd() {
                    Calculator calc = new Calculator();
                    assertEquals(5, calc.add(2, 3));
                }
                
                @Test
                public void testSubtract() {
                    Calculator calc = new Calculator();
                    assertEquals(1, calc.subtract(3, 2));
                }
            }
            """)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertEqual(self.agent.threshold, 0.7)
        self.assertEqual(self.agent.min_confidence, 0.6)
        self.assertEqual(self.agent.device, 'cpu')
    
    def test_fallback_prediction(self):
        """Test fallback prediction when no model is loaded."""
        test_cases = [
            MockTestCase("test1", "test1.java"),
            MockTestCase("test2", "test2.java")
        ]
        
        predictions = self.agent._fallback_prediction(test_cases)
        
        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIn(pred.test_id, ["test1", "test2"])
            self.assertEqual(pred.relevance_score, 0.5)
            self.assertEqual(pred.confidence, 0.5)
            self.assertTrue(pred.predicted_label)
    
    @patch('torch_geometric.data.Data')
    @patch('src.layer4_patch_regression.regression_sentinel_agent.ASTCFGGenerator')
    @patch('src.layer4_patch_regression.regression_sentinel_agent.SemanticDiffAnalyzer')
    def test_predict_test_relevance(self, mock_diff_analyzer, mock_ast_cfg, mock_data):
        """Test the main prediction method."""
        # Setup mocks
        mock_diff_analyzer.return_value.analyze_diff.return_value = MagicMock(
            changed_files={
                'src/main/java/com/example/Calculator.java': [
                    {'start_line': 1, 'end_line': 10}
                ]
            }
        )
        
        # Mock model prediction
        self.agent.model = MagicMock()
        self.agent.model.return_value = torch.tensor([[0.8]])
        
        test_cases = [
            MockTestCase("testAdd", "src/test/java/com/example/CalculatorTest.java"),
            MockTestCase("testSubtract", "src/test/java/com/example/CalculatorTest.java")
        ]
        
        patch_content = """
        diff --git a/src/main/java/com/example/Calculator.java b/src/main/java/com/example/Calculator.java
        index 1234567..89abcde 100644
        --- a/src/main/java/com/example/Calculator.java
        +++ b/src/main/java/com/example/Calculator.java
        @@ -1,5 +1,5 @@
         package com.example;
        -public class Calculator {
        +public class Calculator implements Serializable {
             public int add(int a, int b) {
                 return a + b;
             }
        """
        
        predictions = self.agent.predict_test_relevance(
            patch_content=patch_content,
            test_cases=test_cases,
            project_path=self.project_path
        )
        
        self.assertEqual(len(predictions), 2)
        for pred in predictions:
            self.assertIn(pred.test_id, ["testAdd", "testSubtract"])
            self.assertGreaterEqual(pred.relevance_score, 0.0)
            self.assertLessEqual(pred.relevance_score, 1.0)
            self.assertGreaterEqual(pred.confidence, 0.0)
            self.assertLessEqual(pred.confidence, 1.0)
    
    def test_combine_graphs(self):
        """Test combining change and test graphs."""
        # Create mock graphs
        change_graph = MagicMock()
        change_graph.x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        change_graph.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        
        test_graph = MagicMock()
        test_graph.x = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        test_graph.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        
        # Call the method
        combined = self.agent._combine_graphs(
            {'graph': change_graph},
            test_graph
        )
        
        # Verify the combined graph
        self.assertEqual(combined.x.size(0), 5)  # 2 + 3 nodes
        self.assertEqual(combined.edge_index.size(1), 5)  # 1 + 2 + 2 edges (including connecting edge)
    
    def test_save_and_load_model(self):
        """Test saving and loading the model."""
        # Create a temporary file
        model_path = os.path.join(self.temp_dir, 'test_model.pt')
        
        # Create a mock model
        self.agent.model = GATModel()
        
        # Save the model
        self.agent.save_model(model_path)
        
        # Load the model
        loaded_agent = RegressionSentinelAgent({
            'model_path': model_path,
            'device': 'cpu'
        })
        
        # Verify the model was loaded
        self.assertIsNotNone(loaded_agent.model)
        self.assertIsInstance(loaded_agent.model, GATModel)
    
    def test_gat_model_forward(self):
        """Test the forward pass of the GAT model."""
        # Create a test graph
        x = torch.randn(4, 128)  # 4 nodes, 128 features each
        edge_index = torch.tensor([[0, 1, 2, 0, 3], [1, 2, 3, 3, 0]], dtype=torch.long)
        
        # Initialize the model
        model = GATModel()
        
        # Forward pass
        output = model(x, edge_index)
        
        # Verify output shape
        self.assertEqual(output.shape, (4, 1))  # One output per node
        
        # Verify output is in [0, 1] range (due to sigmoid)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))
    
    def test_select_regression_tests(self):
        """Test the select_regression_tests method."""
        # Create test cases with mock predictions
        test_cases = [
            MockTestCase(f"test_{i}", f"test_{i}.java") for i in range(10)
        ]
        
        # Mock predict_test_relevance to return predefined scores
        with patch.object(self.agent, 'predict_test_relevance') as mock_predict:
            # Create mock predictions with varying relevance scores
            mock_predict.return_value = [
                TestRelevancePrediction(
                    test_id=f"test_{i}",
                    relevance_score=i/10.0,  # Scores from 0.0 to 0.9
                    confidence=0.8,
                    predicted_label=i >= 5,  # Arbitrary threshold
                    explanation=f"Test {i} prediction"
                )
                for i in range(10)
            ]
            
            # Test top-3 selection
            selected = self.agent.select_regression_tests(
                patch_content="mock_patch",
                test_cases=test_cases,
                project_path=self.project_path,
                k=3
            )
            
            # Verify top-3 tests are selected (should be tests 7, 8, 9)
            self.assertEqual(len(selected), 3)
            self.assertEqual(selected[0].test_id, "test_9")
            self.assertEqual(selected[1].test_id, "test_8")
            self.assertEqual(selected[2].test_id, "test_7")
            
            # Test with threshold
            selected = self.agent.select_regression_tests(
                patch_content="mock_patch",
                test_cases=test_cases,
                project_path=self.project_path,
                k=10,  # Request more than available
                threshold=0.5  # Only tests with score >= 0.5
            )
            
            # Should return tests 5-9 (5 tests)
            self.assertEqual(len(selected), 5)
            for test in selected:
                self.assertGreaterEqual(test.relevance_score, 0.5)
    
    def test_select_regression_tests_empty(self):
        """Test select_regression_tests with no tests meeting threshold."""
        test_cases = [
            MockTestCase("test_1", "test_1.java"),
            MockTestCase("test_2", "test_2.java")
        ]
        
        with patch.object(self.agent, 'predict_test_relevance') as mock_predict:
            mock_predict.return_value = [
                TestRelevancePrediction(
                    test_id=test.test_id,
                    relevance_score=0.3,  # Below threshold
                    confidence=0.8,
                    predicted_label=False,
                    explanation="Test prediction"
                )
                for test in test_cases
            ]
            
            # Test with high threshold
            selected = self.agent.select_regression_tests(
                patch_content="mock_patch",
                test_cases=test_cases,
                project_path=self.project_path,
                threshold=0.5  # Higher than test scores
            )
            
            # Should return empty list
            self.assertEqual(len(selected), 0)

if __name__ == '__main__':
    unittest.main()
