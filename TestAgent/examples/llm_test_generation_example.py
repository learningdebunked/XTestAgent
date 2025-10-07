"""
Example usage of LLMTestGenerationAgent for generating and refining test cases.

This script demonstrates how to:
1. Initialize the LLM test generation agent
2. Generate test cases for a Java method
3. Refine tests based on feedback
4. Handle errors and API key management
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.layer2_test_generation.llm_test_agent import (
    LLMTestGenerationAgent,
    GeneratedTest
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_generation.log')
    ]
)
logger = logging.getLogger(__name__)

def get_api_key() -> str:
    """Get API key from environment or user input"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            raise ValueError("API key is required to run this example")
    return api_key

def display_test(test: GeneratedTest, test_num: int = None) -> None:
    """Display test information in a formatted way"""
    if test_num is not None:
        print(f"\n{'='*50}")
        print(f"TEST {test_num}")
        print(f"{'='*50}")
    
    print(f"\nName: {test.test_name}")
    print(f"Confidence: {test.confidence_score:.2f}")
    print(f"Rationale: {test.rationale}")
    print("\nCode:")
    print("-" * 30)
    print(test.test_code)
    print("-" * 30)

def example_test_generation(api_key: str) -> List[GeneratedTest]:
    """
    Example of generating test cases for a Java method.
    
    Args:
        api_key: OpenAI API key
        
    Returns:
        List of generated test cases
    """
    try:
        # Initialize the test generation agent
        logger.info("Initializing LLMTestGenerationAgent...")
        agent = LLMTestGenerationAgent(
            api_key=api_key,
            model="gpt-4",  # or "gpt-3.5-turbo" for faster/cheaper testing
            temperature=0.7
        )
        
        # Define the method to test
        method_signature = "public int divide(int a, int b)"
        method_source = """
public int divide(int a, int b) {
    if (b == 0) throw new IllegalArgumentException("Division by zero");
    return a / b;
}
"""
        # Context about the code changes
        context = {
            'commit_message': 'Fix division by zero',
            'textual_diff': """-     return a / b;  // No zero check
+     if (b == 0) throw new IllegalArgumentException(\"Division by zero\");
+     return a / b;  // With zero check""",
            'modified_methods': ['divide']
        }
        
        # Generate test cases
        logger.info("Generating test cases...")
        tests = agent.generate_tests(
            method_signature=method_signature,
            method_source=method_source,
            semantic_context=context,
            num_tests=3  # Generate 3 test cases
        )
        
        logger.info(f"Successfully generated {len(tests)} test cases")
        return tests
        
    except Exception as e:
        logger.error(f"Error in test generation: {str(e)}")
        raise

def example_test_refinement(agent: LLMTestGenerationAgent, 
                          test: GeneratedTest) -> GeneratedTest:
    """
    Example of refining a test case based on feedback.
    
    Args:
        agent: Initialized LLMTestGenerationAgent
        test: Test case to refine
        
    Returns:
        Refined test case
    """
    try:
        logger.info(f"Refining test: {test.test_name}")
        
        # Example feedback for refinement
        feedback = """
        The test should include:
        1. More descriptive assertion messages
        2. Test case for division by zero
        3. Test case with negative numbers
        4. Edge case with minimum integer value
        """
        
        # Refine the test
        refined_test = agent.refine_test(test, feedback)
        
        logger.info(f"Successfully refined test: {refined_test.test_name}")
        return refined_test
        
    except Exception as e:
        logger.error(f"Error in test refinement: {str(e)}")
        return test  # Return original test if refinement fails

def main():
    """Main function to run the example"""
    try:
        # Get API key
        api_key = get_api_key()
        
        # Example 1: Generate tests
        print("\n" + "="*50)
        print("GENERATING TEST CASES")
        print("="*50)
        tests = example_test_generation(api_key)
        
        # Display generated tests
        for i, test in enumerate(tests, 1):
            display_test(test, i)
        
        # Example 2: Refine a test
        if tests:
            print("\n" + "="*50)
            print("REFINING A TEST CASE")
            print("="*50)
            
            # Take the first test for refinement
            test_to_refine = tests[0]
            print("\nOriginal test:")
            display_test(test_to_refine)
            
            # Refine the test
            refined_test = example_test_refinement(
                LLMTestGenerationAgent(api_key=api_key),
                test_to_refine
            )
            
            print("\nRefined test:")
            display_test(refined_test)
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        logger.critical(f"Example failed: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
