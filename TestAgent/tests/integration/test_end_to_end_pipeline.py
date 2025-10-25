"""
End-to-End Integration Tests for TestAgentX

Tests the complete pipeline from code input to test generation,
validation, and patch verification.
"""

import unittest
import sys
from pathlib import Path
import tempfile
import shutil
import os

# Fix for macOS multiprocessing issues with PyTorch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Set multiprocessing start method to 'spawn' for macOS compatibility
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from layer1_preprocessing.code_encoder import CodeEncoder
from layer1_preprocessing.ast_cfg_generator import ASTCFGGenerator
from layer2_test_generation.llm_test_agent import LLMTestGenerationAgent
from layer3_fuzzy_validation.fuzzy_assertion_agent import FuzzyAssertionAgent
from layer4_patch_regression.patch_verification_agent import PatchVerificationAgent
from config.config_loader import ConfigLoader, Config


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end integration tests"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test fixtures"""
        cls.config_loader = ConfigLoader()
        cls.config = cls.config_loader.load()
        
        # Create temporary directory for test outputs
        cls.temp_dir = tempfile.mkdtemp()
        
        # Sample Java code
        cls.sample_java_code = """
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int subtract(int a, int b) {
        return a - b;
    }
    
    public int multiply(int a, int b) {
        return a * b;
    }
    
    public int divide(int a, int b) {
        if (b == 0) {
            throw new IllegalArgumentException("Division by zero");
        }
        return a / b;
    }
}
"""
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup test fixtures"""
        if Path(cls.temp_dir).exists():
            shutil.rmtree(cls.temp_dir)
    
    def test_01_code_preprocessing(self):
        """Test Layer 1: Code preprocessing and encoding"""
        print("\n=== Testing Layer 1: Code Preprocessing ===")
        
        # Test AST/CFG generation
        ast_generator = ASTCFGGenerator()
        
        # Write sample code to temp file
        java_file = Path(self.temp_dir) / "Calculator.java"
        with open(java_file, 'w') as f:
            f.write(self.sample_java_code)
        
        # Parse Java file
        methods = ast_generator.parse_java_file(java_file)
        
        self.assertGreater(len(methods), 0, "Should extract methods from Java file")
        self.assertTrue(any(m.method_name == 'add' for m in methods), "Should find 'add' method")
        
        print(f"✓ Extracted {len(methods)} methods")
        
        # Test code encoding
        encoder = CodeEncoder(
            model_name=self.config.get('code_encoder.model_name', 'microsoft/codebert-base')
        )
        
        method = next(m for m in methods if m.method_name == 'add')
        embedding = encoder.encode_method(
            method_signature=method.signature,
            method_body=method.source_code
        )
        
        self.assertIsNotNone(embedding, "Should generate embedding")
        self.assertEqual(len(embedding.embedding), 768, "Embedding should be 768-dim")
        
        print(f"✓ Generated embedding: {embedding.embedding.shape}")
    
    def test_02_test_generation(self):
        """Test Layer 2: Test generation"""
        print("\n=== Testing Layer 2: Test Generation ===")
        
        # Initialize test generation agent
        agent = LLMTestGenerationAgent(
            model_name=self.config.get('test_generation.llm.model_name', 'gpt-4')
        )
        
        # Generate tests for add method
        tests = agent.generate_tests(
            method_signature="public int add(int a, int b)",
            method_source="public int add(int a, int b) { return a + b; }",
            semantic_context={'class_name': 'Calculator'},
            num_tests=3
        )
        
        self.assertGreater(len(tests), 0, "Should generate tests")
        
        for test in tests:
            self.assertIn('test_name', test, "Test should have name")
            self.assertIn('test_code', test, "Test should have code")
            print(f"✓ Generated test: {test['test_name']}")
    
    def test_03_fuzzy_validation(self):
        """Test Layer 3: Fuzzy validation"""
        print("\n=== Testing Layer 3: Fuzzy Validation ===")
        
        # Initialize fuzzy assertion agent
        validator = FuzzyAssertionAgent(
            threshold=self.config.get('fuzzy_validation.threshold', 0.7)
        )
        
        # Test validation
        output_buggy = "Result: 5"
        output_fixed = "Result: 5.0"
        
        result = validator.validate_output(
            output_buggy=output_buggy,
            output_fixed=output_fixed
        )
        
        self.assertIsNotNone(result, "Should return validation result")
        self.assertIn('is_valid', result.__dict__, "Result should have is_valid field")
        self.assertIn('confidence', result.__dict__, "Result should have confidence field")
        
        print(f"✓ Validation result: valid={result.is_valid}, confidence={result.confidence:.2f}")
    
    def test_04_patch_verification(self):
        """Test Layer 4: Patch verification"""
        print("\n=== Testing Layer 4: Patch Verification ===")
        
        # Initialize patch verification agent
        verifier = PatchVerificationAgent(
            epsilon=self.config.get('patch_verification.epsilon', 0.1)
        )
        
        # Create mock test cases
        test_cases = [
            {
                'id': 'test1',
                'class_name': 'CalculatorTest',
                'method_name': 'testAdd'
            }
        ]
        
        # Note: This would require actual project setup
        # For now, we test the interface
        self.assertIsNotNone(verifier, "Verifier should initialize")
        print("✓ Patch verifier initialized")
    
    def test_05_complete_pipeline(self):
        """Test complete pipeline integration"""
        print("\n=== Testing Complete Pipeline ===")
        
        # 1. Preprocess code
        ast_generator = ASTCFGGenerator()
        java_file = Path(self.temp_dir) / "Calculator.java"
        with open(java_file, 'w') as f:
            f.write(self.sample_java_code)
        
        methods = ast_generator.parse_java_file(java_file)
        print(f"✓ Step 1: Extracted {len(methods)} methods")
        
        # 2. Encode method
        encoder = CodeEncoder()
        method = methods[0]
        embedding = encoder.encode_method(
            method_signature=method.signature,
            method_body=method.source_code
        )
        print(f"✓ Step 2: Generated embedding")
        
        # 3. Generate tests
        agent = LLMTestGenerationAgent()
        tests = agent.generate_tests(
            method_signature=method.signature,
            method_source=method.source_code,
            num_tests=2
        )
        print(f"✓ Step 3: Generated {len(tests)} tests")
        
        # 4. Validate outputs
        validator = FuzzyAssertionAgent()
        validation_result = validator.validate_output(
            output_buggy="5",
            output_fixed="5"
        )
        print(f"✓ Step 4: Validated outputs (confidence={validation_result.confidence:.2f})")
        
        # 5. Verify patch (interface test)
        verifier = PatchVerificationAgent()
        print(f"✓ Step 5: Patch verifier ready")
        
        print("\n✅ Complete pipeline test passed!")


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration system integration"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        print("\n=== Testing Configuration System ===")
        
        loader = ConfigLoader()
        config = loader.load()
        
        # Test accessing config values
        embedding_dim = config.get('code_encoder.embedding_dim')
        self.assertIsNotNone(embedding_dim, "Should load embedding_dim")
        print(f"✓ Loaded embedding_dim: {embedding_dim}")
        
        temperature = config.get('test_generation.llm.temperature')
        self.assertIsNotNone(temperature, "Should load temperature")
        print(f"✓ Loaded temperature: {temperature}")
        
        threshold = config.get('fuzzy_validation.threshold')
        self.assertIsNotNone(threshold, "Should load threshold")
        print(f"✓ Loaded threshold: {threshold}")
    
    def test_config_override(self):
        """Test configuration override"""
        print("\n=== Testing Configuration Override ===")
        
        loader = ConfigLoader()
        config = loader.load()
        
        # Override value
        config.set('test_generation.llm.temperature', 0.9)
        new_temp = config.get('test_generation.llm.temperature')
        
        self.assertEqual(new_temp, 0.9, "Should override temperature")
        print(f"✓ Overridden temperature: {new_temp}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery"""
    
    def test_retry_mechanism(self):
        """Test retry on error"""
        print("\n=== Testing Error Handling ===")
        
        from utils.error_handling import retry_on_error
        
        attempt_count = [0]
        
        @retry_on_error(max_retries=3, delay=0.1)
        def flaky_function():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise Exception("Temporary failure")
            return "Success"
        
        result = flaky_function()
        
        self.assertEqual(result, "Success", "Should succeed after retries")
        self.assertEqual(attempt_count[0], 3, "Should retry 3 times")
        print(f"✓ Retry mechanism worked ({attempt_count[0]} attempts)")
    
    def test_error_recovery(self):
        """Test error recovery strategies"""
        print("\n=== Testing Error Recovery ===")
        
        from utils.error_handling import ErrorRecovery
        
        # Test fallback chain
        def primary():
            raise Exception("Primary failed")
        
        def fallback():
            return "Fallback success"
        
        result = ErrorRecovery.fallback_chain(primary, fallback)
        
        self.assertEqual(result, "Fallback success", "Should use fallback")
        print("✓ Fallback chain worked")


def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigurationIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
