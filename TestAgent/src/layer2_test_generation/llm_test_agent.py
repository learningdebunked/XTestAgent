"""
Implements Equation (3) from Section 3.4.1:
T_gen = LLM_test(s_vec_i, M, Delta_sem)

LLM-based test case generation using prompt engineering.
"""

import os
import json
import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import openai
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GeneratedTest:
    """Represents a generated test case"""
    test_code: str
    test_name: str
    method_under_test: str
    assertions: List[str]
    confidence_score: float
    rationale: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'test_code': self.test_code,
            'test_name': self.test_name,
            'method_under_test': self.method_under_test,
            'assertions': self.assertions,
            'confidence_score': self.confidence_score,
            'rationale': self.rationale,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneratedTest':
        """Create from dictionary"""
        return cls(
            test_code=data['test_code'],
            test_name=data['test_name'],
            method_under_test=data['method_under_test'],
            assertions=data['assertions'],
            confidence_score=data['confidence_score'],
            rationale=data['rationale'],
            metadata=data.get('metadata', {})
        )

class LLMTestGenerationAgent:
    """
    Generates test cases using Large Language Models.
    Implements Equation (3): T_gen = LLM_test(s_vec, M, Delta_sem)
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "gpt-4",
                 temperature: float = 0.7,
                 max_tokens: int = 2000):
        """
        Initialize LLM test generation agent.
        
        Args:
            api_key: OpenAI API key (reads from env if None)
            model: Model to use for generation
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in OPENAI_API_KEY environment variable"
            )
            
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Load prompt templates
        self.prompts = self._load_prompt_templates()
        
        logger.info(f"Initialized LLMTestGenerationAgent with model: {model}")
    
    def refine_test(self, test: GeneratedTest, 
                   feedback: str) -> GeneratedTest:
        """
        Refine a test based on feedback.
        Supports human-in-the-loop improvement.
        
        Args:
            test: The original GeneratedTest to refine
            feedback: Feedback for improving the test
            
        Returns:
            Refined GeneratedTest object
        """
        logger.info(f"Refining test '{test.test_name}' based on feedback")
        
        refine_prompt = f"""
You are a senior test engineer. Please refine the following JUnit test based on the provided feedback.

ORIGINAL TEST:
```java
{test.test_code}
```

FEEDBACK TO ADDRESS:
{feedback}

Please provide an improved version of this test that addresses the feedback. 
Include only the test method code in your response, without additional explanation.

IMPROVED TEST:
```java
"""
        
        try:
            # Call LLM for refinement
            response = self._call_llm(refine_prompt)
            
            # Extract the code block from the response
            if '```java' in response:
                refined_code = response.split('```java')[1].split('```')[0].strip()
            else:
                refined_code = response.strip()
            
            # Update the test with refined code
            refined_test = GeneratedTest(
                test_code=refined_code,
                test_name=test.test_name,
                method_under_test=test.method_under_test,
                assertions=test.assertions,  # Will be updated below
                confidence_score=min(1.0, test.confidence_score + 0.1),  # Slightly increase confidence
                rationale=f"Refined based on feedback: {feedback}",
                metadata={
                    **test.metadata,
                    'refinement': True,
                    'original_test': test.test_code,
                    'feedback': feedback
                }
            )
            
            # Update assertions from refined code
            refined_test.assertions = [
                line.strip() 
                for line in refined_code.split('\n') 
                if any(a in line for a in ['assert', 'verify', 'expect'])
            ]
            
            logger.info(f"Successfully refined test '{test.test_name}'")
            return refined_test
            
        except Exception as e:
            logger.error(f"Error refining test: {str(e)}")
            # Return original test if refinement fails
            return test

    def _load_prompt_templates(self) -> Dict[str, Any]:
        """Load prompt templates from file or use defaults"""
        # Default templates
        return {
            'test_generation_template': """
            You are a senior software engineer specializing in writing high-quality test cases.
            Generate JUnit test cases for the given Java method based on the provided context.
            
            Guidelines:
            1. Focus on testing edge cases and boundary conditions
            2. Include tests for the bug fix described in the commit message
            3. Ensure 100% code coverage of the method
            4. Include both positive and negative test cases
            5. Add assertions to verify expected behavior
            6. Include clear test names that describe the scenario
            7. Handle potential exceptions appropriately
            
            Return the test cases in the following JSON format:
            {
                "tests": [
                    {
                        "test_name": "test_<scenario>",
                        "test_code": "@Test\npublic void test_<scenario>() { ... }",
                        "assertions": ["assertion 1", "assertion 2"],
                        "confidence_score": <0.0-1.0>,
                        "rationale": "Explanation of what this test verifies"
                    }
                ]
            }
            """,
            'few_shot_examples': [
                {
                    "method_signature": "public int divide(int a, int b)",
                    "method_source": "public int divide(int a, int b) {\n    if (b == 0) throw new IllegalArgumentException(\"Cannot divide by zero\");\n    return a / b;\n}",
                    "commit_message": "Fix division by zero in Calculator.divide",
                    "semantic_diff": "-     return a / b;  // No zero check\n+     if (b == 0) throw new IllegalArgumentException(\"Cannot divide by zero\");\n+     return a / b;  // With zero check",
                    "generated_tests": [
                        {
                            "test_name": "test_divide_positive_numbers",
                            "test_code": "@Test\npublic void test_divide_positive_numbers() {\n    Calculator calc = new Calculator();\n    int result = calc.divide(10, 2);\n    assertEquals(5, result);\n}",
                            "assertions": ["assertEquals(5, result)"],
                            "confidence_score": 0.95,
                            "rationale": "Test normal division of positive numbers"
                        },
                        {
                            "test_name": "test_divide_by_zero_throws_exception",
                            "test_code": "@Test(expected = IllegalArgumentException.class)\npublic void test_divide_by_zero_throws_exception() {\n    Calculator calc = new Calculator();\n    calc.divide(10, 0);\n}",
                            "assertions": ["Expected IllegalArgumentException"],
                            "confidence_score": 1.0,
                            "rationale": "Verify division by zero throws IllegalArgumentException"
                        }
                    ]
                }
            ]
        }
    
    def generate_tests(self, method_signature: str,
                      method_source: str,
                      semantic_context: Dict[str, Any],
                      num_tests: int = 5) -> List[GeneratedTest]:
        """
        Generate test cases for a given method.
        
        Implements Equation (3) from paper: T_gen = LLM_test(s_vec, M, Delta_sem)
        
        Args:
            method_signature: Signature of method to test
            method_source: Source code of method
            semantic_context: Context from semantic diff analysis
            num_tests: Number of tests to generate
            
        Returns:
            List of GeneratedTest objects
        """
        try:
            logger.info(f"Generating {num_tests} test cases for method: {method_signature}")
            
            # Build prompt with context
            prompt = self._build_prompt(
                method_signature=method_signature,
                method_source=method_source,
                semantic_context=semantic_context,
                num_tests=num_tests
            )
            
            # Call LLM
            logger.debug("Calling LLM for test generation...")
            response = self._call_llm(prompt)
            
            # Parse response into test cases
            tests = self._parse_test_response(
                response=response,
                method_signature=method_signature,
                semantic_context=semantic_context
            )
            
            logger.info(f"Successfully generated {len(tests)} test cases")
            return tests
            
        except Exception as e:
            logger.error(f"Error generating tests: {str(e)}")
            raise
    
    def _build_prompt(self, method_signature: str, 
                     method_source: str,
                     semantic_context: Dict[str, Any],
                     num_tests: int) -> str:
        """
        Build few-shot prompt for test generation.
        
        Corresponds to prompt engineering described in Section 3.4.1.
        """
        template = self.prompts['test_generation_template']
        examples = self.prompts['few_shot_examples']
        
        # Format examples
        example_strs = []
        for ex in examples[:2]:  # Use first 2 examples
            ex_tests = json.dumps({"tests": ex["generated_tests"]}, indent=2)
            example_strs.append(
                f"Method: {ex['method_signature']}\n"
                f"Source Code:\n{ex['method_source']}\n"
                f"Commit Message: {ex['commit_message']}\n"
                f"Semantic Diff:\n{ex['semantic_diff']}\n"
                f"Generated Tests:\n{ex_tests}\n"
            )
        
        prompt = f"""{template}

## Example 1
{example_strs[0]}

## Example 2
{example_strs[1]}

## Your Task
Generate {num_tests} test cases for the following method:

Method: {method_signature}

Source Code:
{method_source}

Commit Message: {commit_message}

Semantic Diff:
{semantic_diff}

Modified Methods: {modified_methods}

Please generate {num_tests} test cases in the specified JSON format.
""".format(
            template=template,
            example_strs=example_strs,
            num_tests=num_tests,
            method_signature=method_signature,
            method_source=method_source,
            commit_message=semantic_context.get('commit_message', 'No commit message'),
            semantic_diff=semantic_context.get('textual_diff', 'No diff available'),
            modified_methods=', '.join(semantic_context.get('modified_methods', ['Unknown']))
        )
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the language model to generate test cases.
        
        Args:
            prompt: Formatted prompt string
            
        Returns:
            Raw response from the LLM
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates high-quality test cases for Java methods."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"}
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise
    
    def _parse_test_response(self, response: str, 
                           method_signature: str,
                           semantic_context: Dict[str, Any]) -> List[GeneratedTest]:
        """
        Parse LLM response into GeneratedTest objects.
        
        Args:
            response: Raw response from LLM
            method_signature: Signature of method being tested
            semantic_context: Context from semantic diff analysis
            
        Returns:
            List of GeneratedTest objects
        """
        try:
            # Parse JSON response
            response_data = json.loads(response)
            tests_data = response_data.get('tests', [])
            
            # Convert to GeneratedTest objects
            tests = []
            for test_data in tests_data:
                try:
                    test = GeneratedTest(
                        test_code=test_data['test_code'],
                        test_name=test_data['test_name'],
                        method_under_test=method_signature,
                        assertions=test_data.get('assertions', []),
                        confidence_score=float(test_data.get('confidence_score', 0.8)),
                        rationale=test_data.get('rationale', ''),
                        metadata={
                            'model': self.model,
                            'semantic_context': semantic_context
                        }
                    )
                    tests.append(test)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Skipping invalid test case: {str(e)}")
                    continue
            
            return tests
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            # Try to extract code blocks if JSON parsing fails
            return self._fallback_parse(response, method_signature)
    
    def _fallback_parse(self, response: str, 
                       method_signature: str) -> List[GeneratedTest]:
        """
        Fallback parsing when JSON parsing fails.
        Extracts test methods using regex patterns.
        """
        tests = []
        
        # Pattern to match test methods
        test_pattern = r'(@Test[^}]*?public\s+void\s+(test\w+)\s*\([^)]*\)\s*\{[^}]*\})'
        matches = re.finditer(test_pattern, response, re.DOTALL)
        
        for match in matches:
            test_code = match.group(1)
            test_name = match.group(2)
            
            # Extract assertions
            assertions = re.findall(r'assert\w+\s*\([^;]+;', test_code)
            
            test = GeneratedTest(
                test_code=test_code,
                test_name=test_name,
                method_under_test=method_signature,
                assertions=assertions,
                confidence_score=0.7,  # Lower confidence for fallback parsing
                rationale="Generated using fallback parser",
                metadata={
                    'model': self.model,
                    'parsing': 'fallback_regex'
                }
            )
            tests.append(test)
        
        return tests
    
    def save_tests(self, tests: List[GeneratedTest], 
                  output_dir: Path) -> List[Path]:
        """
        Save generated tests to files.
        
        Args:
            tests: List of GeneratedTest objects
            output_dir: Directory to save test files
            
        Returns:
            List of paths to saved test files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        # Group tests by method
        tests_by_method: Dict[str, List[GeneratedTest]] = {}
        for test in tests:
            method = test.method_under_test
            if method not in tests_by_method:
                tests_by_method[method] = []
            tests_by_method[method].append(test)
        
        # Save tests for each method
        for method, method_tests in tests_by_method.items():
            # Generate class name from method signature
            method_name = method.split('(')[0].split()[-1]
            class_name = f"{method_name}Test"
            
            # Combine test code
            test_code = f"""/*
 * Generated test class for: {method}
 * Generated by LLMTestGenerationAgent
 */

import org.junit.Test;
import static org.junit.Assert.*;

public class {class_name} {{

{test_methods}

}}"""
            
            # Add test methods
            test_methods = []
            for test in method_tests:
                test_methods.append(f"    // {test.rationale}")
                test_methods.append(f"    {test.test_code}\n")
            
            # Format final code
            test_code = test_code.format(
                class_name=class_name,
                test_methods='\n'.join(test_methods)
            )
            
            # Write to file
            file_path = output_dir / f"{class_name}.java"
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(test_code)
            
            saved_paths.append(file_path)
            logger.info(f"Saved {len(method_tests)} tests to {file_path}")
        
        return saved_paths

def example_usage() -> None:
    """Example usage of the LLMTestGenerationAgent"""
    import time
    
    print("Starting LLM test generation example...")
    start_time = time.time()
    
    # Initialize agent
    agent = LLMTestGenerationAgent()
    
    # Example method to test
    method_signature = "public int divide(int a, int b)"
    method_source = """public int divide(int a, int b) {
    if (b == 0) throw new IllegalArgumentException("Cannot divide by zero");
    return a / b;
}"""
    
    # Semantic context (from diff analysis)
    semantic_context = {
        'commit_message': 'Fix division by zero in Calculator.divide',
        'textual_diff': """-     return a / b;  // No zero check\n+     if (b == 0) throw new IllegalArgumentException(\"Cannot divide by zero\");\n+     return a / b;  // With zero check""",
        'modified_methods': ['divide']
    }
    
    # Generate tests
    print(f"\nGenerating tests for method: {method_signature}")
    tests = agent.generate_tests(
        method_signature=method_signature,
        method_source=method_source,
        semantic_context=semantic_context,
        num_tests=3
    )
    
    # Print results
    print(f"\nGenerated {len(tests)} test cases:")
    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}: {test.test_name}")
        print(f"Confidence: {test.confidence_score:.2f}")
        print(f"Rationale: {test.rationale}")
        print(f"Code:\n{test.test_code}")
    
    # Save tests to file
    output_dir = Path("generated_tests")
    saved_paths = agent.save_tests(tests, output_dir)
    print(f"\nSaved tests to: {', '.join(str(p) for p in saved_paths)}")
    
    print(f"\nTest generation completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    example_usage()
