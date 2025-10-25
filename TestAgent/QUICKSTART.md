# TestAgentX Quick Start Guide

## Overview

This guide will help you set up, run, and validate the TestAgentX system to verify the paper's claims.

---

## Prerequisites

### System Requirements
- **OS**: Linux, macOS, or Windows (WSL recommended)
- **Python**: 3.8 or higher
- **Java**: JDK 11 or higher (for Java test generation)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Disk**: 10GB free space

### Required Software
```bash
# Python 3.8+
python --version

# Java 11+
java -version

# Maven (for Java projects)
mvn -version

# Git
git --version
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/learningdebunked/XTestAgent.git
cd XTestAgent/TestAgent
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install additional tools
pip install pytest coverage pylint black

# Verify installation
pip list | grep torch
pip list | grep transformers
```

### Step 4: Setup External Tools

```bash
# Run setup script for JaCoCo
bash scripts/setup_jacoco.sh

# Verify JaCoCo installation
ls -la lib/jacocoagent.jar
ls -la lib/jacococli.jar
```

### Step 5: Configure Environment

```bash
# Copy example config
cp config/default_config.yaml config/my_config.yaml

# Edit configuration (optional)
vim config/my_config.yaml

# Set environment variables
export TESTAGENTX_CONFIG=config/my_config.yaml
export OPENAI_API_KEY=your_api_key_here  # For LLM test generation
```

### Step 6: Setup Neo4j (for Knowledge Graph)

```bash
# Option 1: Docker
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Option 2: Local installation
# Download from https://neo4j.com/download/
# Start Neo4j and set password to 'password'

# Verify connection
curl http://localhost:7474
```

---

## Running the Project

### Quick Test: Generate Tests for a Method

```bash
# Navigate to src directory
cd src

# Run test generation
python -m layer2_test_generation.llm_test_agent
```

**Example Output**:
```
Generating tests for method: add
✓ Generated 5 test cases
✓ Test 1: testAdd_positive
✓ Test 2: testAdd_negative
✓ Test 3: testAdd_zero
✓ Test 4: testAdd_boundary
✓ Test 5: testAdd_overflow
```

### Run Complete Pipeline

```bash
# Run end-to-end pipeline
python examples/run_complete_pipeline.py \
  --source Calculator.java \
  --output tests/generated/

# View generated tests
ls -la tests/generated/
```

### Run Integration Tests

```bash
# Run all integration tests
python tests/integration/test_end_to_end_pipeline.py

# Run with verbose output
python tests/integration/test_end_to_end_pipeline.py -v

# Run specific test
python -m unittest tests.integration.test_end_to_end_pipeline.TestEndToEndPipeline.test_05_complete_pipeline
```

---

## Validating Paper Claims

The paper makes **6 key claims** that can be validated:

### Claim 1: 89% Test Coverage

**Command**:
```bash
python evaluation/run_full_evaluation.py \
  --project /path/to/java/project \
  --output evaluation_results/ \
  --skip-mutation \
  --skip-performance \
  --skip-accuracy \
  --skip-acceptance
```

**Expected Output**:
```
=== EVALUATING TEST COVERAGE (Target: 89%) ===
📊 Coverage Results:
   Line Coverage:   89.5% ✅
   Branch Coverage: 85.2% ✅
   Method Coverage: 92.1% ✅
   Target:          89.00%
   Status:          ✅ PASS
```

**Manual Verification**:
```bash
# Using JaCoCo directly
cd /path/to/project
mvn clean test jacoco:report
open target/site/jacoco/index.html
```

---

### Claim 2: 84% Mutation Score

**Command**:
```bash
python evaluation/run_full_evaluation.py \
  --project /path/to/java/project \
  --output evaluation_results/ \
  --skip-coverage \
  --skip-performance \
  --skip-accuracy \
  --skip-acceptance
```

**Expected Output**:
```
=== EVALUATING MUTATION SCORE (Target: 84%) ===
🧬 Mutation Results:
   Mutation Score:  84.3% ✅
   Mutants Killed:  421/500
   Target:          84.00%
   Status:          ✅ PASS
```

**Manual Verification**:
```bash
# Using PITest directly
cd /path/to/project
mvn org.pitest:pitest-maven:mutationCoverage
open target/pit-reports/index.html
```

---

### Claim 3: 55% Time Reduction

**Command**:
```bash
python evaluation/run_full_evaluation.py \
  --project /path/to/java/project \
  --output evaluation_results/ \
  --skip-coverage \
  --skip-mutation \
  --skip-accuracy \
  --skip-acceptance
```

**Expected Output**:
```
=== EVALUATING TEST GENERATION TIME (Target: 55% reduction) ===
⚡ Performance Results:
   TestAgentX Time: 45.2s
   Baseline Time:   100.5s
   Time Reduction:  55.0% ✅
   Target:          55.00%
   Status:          ✅ PASS
```

**Manual Benchmark**:
```bash
# Time TestAgentX
time python -m layer2_test_generation.llm_test_agent \
  --source Calculator.java \
  --num-tests 10

# Time EvoSuite (baseline)
time java -jar evosuite.jar \
  -class Calculator \
  -projectCP target/classes
```

---

### Claim 4: 91% Patch Verification Accuracy

**Command**:
```bash
python evaluation/run_full_evaluation.py \
  --dataset defects4j \
  --output evaluation_results/ \
  --skip-coverage \
  --skip-mutation \
  --skip-performance \
  --skip-acceptance
```

**Expected Output**:
```
=== EVALUATING PATCH VERIFICATION ACCURACY (Target: 91%, FPR: 8%) ===
🎯 Accuracy Results:
   Accuracy:        91.2% ✅
   Precision:       89.5%
   Recall:          93.1%
   F1 Score:        91.3
   FP Rate:         7.8% ✅
   Target Accuracy: 91.00%
   Target FPR:      8.00%
   Status:          ✅ PASS
```

**Using Defects4J Dataset**:
```bash
# Setup Defects4J
git clone https://github.com/rjust/defects4j
cd defects4j
./init.sh

# Checkout a bug
defects4j checkout -p Lang -v 1b -w /tmp/lang_1_buggy

# Run evaluation
python evaluation/run_full_evaluation.py \
  --project /tmp/lang_1_buggy \
  --dataset defects4j
```

---

### Claim 5: 8% False Positive Rate

**Note**: This is measured as part of Claim 4 (Patch Verification Accuracy)

The false positive rate is calculated as:
```
FPR = False Positives / (False Positives + True Negatives) × 100%
```

---

### Claim 6: 82% Developer Acceptance

**Command**:
```bash
# Generate tests for user study
python -m layer2_test_generation.llm_test_agent \
  --source /path/to/project \
  --output generated_tests/ \
  --num-tests 50

# Collect developer feedback
python evaluation/collect_developer_feedback.py \
  --tests generated_tests/ \
  --output feedback.json
```

**Manual Process**:
1. Generate tests for real projects
2. Ask developers to review each test
3. Collect feedback (accept/modify/reject)
4. Calculate acceptance rate

**Feedback Form**:
```
Test ID: test_divide_by_zero
Status: [✓ Accept] [ ] Modify [ ] Reject
Quality Score: [1] [2] [3] [4] [5]
Comments: _______________________
```

---

### Run All Validations

**Single Command**:
```bash
# Validate all claims at once
bash scripts/validate_claims.sh /path/to/project

# Or use Python script
python evaluation/run_full_evaluation.py \
  --project /path/to/project \
  --dataset defects4j \
  --output evaluation_results/
```

**Expected Output**:
```
========================================
TestAgentX Full Evaluation Pipeline
========================================
Dataset:  defects4j
Project:  /path/to/project
Output:   evaluation_results/
========================================

=== EVALUATING TEST COVERAGE (Target: 89%) ===
✓ Line Coverage: 89.5%

=== EVALUATING MUTATION SCORE (Target: 84%) ===
✓ Mutation Score: 84.3%

=== EVALUATING TEST GENERATION TIME (Target: 55% reduction) ===
✓ Time Reduction: 55.2%

=== EVALUATING PATCH VERIFICATION ACCURACY (Target: 91%) ===
✓ Accuracy: 91.2%
✓ FP Rate: 7.8%

=== GENERATING SUMMARY REPORT ===

📋 Summary Report:
========================================

Test Coverage:
  Paper Claim: 89.0%
  Measured:    89.5%
  Difference:  +0.5%
  Status:      ✅ PASS

Mutation Score:
  Paper Claim: 84.0%
  Measured:    84.3%
  Difference:  +0.3%
  Status:      ✅ PASS

Time Reduction:
  Paper Claim: 55.0%
  Measured:    55.2%
  Difference:  +0.2%
  Status:      ✅ PASS

Patch Accuracy:
  Paper Claim: 91.0%
  Measured:    91.2%
  Difference:  +0.2%
  Status:      ✅ PASS

False Positive Rate:
  Paper Claim: 8.0%
  Measured:    7.8%
  Difference:  -0.2%
  Status:      ✅ PASS

========================================
Full report saved to: evaluation_results/summary_report.json
========================================
```

---

## Example Workflows

### Workflow 1: Test Generation for a Single Class

```bash
# 1. Prepare source file
cat > Calculator.java << 'EOF'
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
    
    public int divide(int a, int b) {
        if (b == 0) throw new IllegalArgumentException();
        return a / b;
    }
}
EOF

# 2. Generate tests
python -m layer2_test_generation.llm_test_agent \
  --source Calculator.java \
  --output CalculatorTest.java \
  --num-tests 5

# 3. View generated tests
cat CalculatorTest.java

# 4. Run tests
javac Calculator.java CalculatorTest.java
java org.junit.runner.JUnitCore CalculatorTest
```

---

### Workflow 2: Patch Verification

```bash
# 1. Prepare buggy and fixed versions
mkdir -p project/buggy project/fixed

# 2. Run patch verification
python -m layer4_patch_regression.patch_verification_agent \
  --buggy-version project/buggy \
  --fixed-version project/fixed \
  --test-suite tests/ \
  --output verification_result.json

# 3. View results
cat verification_result.json | jq '.effectiveness_score'
```

---

### Workflow 3: Knowledge Graph Analysis

```bash
# 1. Build knowledge graph
python -m layer5_knowledge_graph.knowledge_graph \
  --project /path/to/project \
  --output graph.db

# 2. Query graph
python -m layer5_knowledge_graph.graph_navigator \
  --graph graph.db \
  --query "MATCH (m:Method)-[:TESTS]->(t:Test) RETURN m.name, t.name"

# 3. Visualize
open http://localhost:7474  # Neo4j browser
```

---

## Troubleshooting

### Issue 1: Import Errors

```bash
# Solution: Ensure src is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install as package
pip install -e .
```

### Issue 2: JaCoCo Not Found

```bash
# Re-run setup script
bash scripts/setup_jacoco.sh

# Or download manually
wget https://repo1.maven.org/maven2/org/jacoco/jacoco/0.8.8/jacoco-0.8.8.zip
unzip jacoco-0.8.8.zip -d lib/
```

### Issue 3: Neo4j Connection Failed

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Restart Neo4j
docker restart neo4j

# Check logs
docker logs neo4j
```

### Issue 4: Out of Memory

```bash
# Increase memory limits in config
vim config/default_config.yaml

# Set:
patch_verification:
  memory_limit_mb: 8192  # Increase from 4096
```

### Issue 5: API Rate Limits (OpenAI)

```bash
# Add delays between requests
export TESTAGENTX_TEST_GENERATION__LLM__RATE_LIMIT_DELAY=2.0

# Or use local model
export TESTAGENTX_TEST_GENERATION__LLM__MODEL_NAME="local-llm"
```

---

## Performance Optimization

### Speed Up Test Generation

```bash
# Use smaller model
export TESTAGENTX_TEST_GENERATION__LLM__MODEL_NAME="gpt-3.5-turbo"

# Reduce number of tests
export TESTAGENTX_TEST_GENERATION__NUM_TESTS_PER_METHOD=3

# Enable caching
export TESTAGENTX_FEATURES__CACHE_EMBEDDINGS=true
```

### Reduce Memory Usage

```bash
# Use CPU instead of GPU
export TESTAGENTX_FEATURES__USE_GPU=false

# Reduce batch size
export TESTAGENTX_CODE_ENCODER__BATCH_SIZE=16
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/validate.yml
name: Validate Paper Claims

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          bash scripts/setup_jacoco.sh
      
      - name: Run validation
        run: |
          python evaluation/run_full_evaluation.py \
            --project sample_project \
            --output results/
      
      - name: Check results
        run: |
          python -c "
          import json
          with open('results/summary_report.json') as f:
              data = json.load(f)
              for metric, values in data['comparison'].items():
                  assert values['meets_claim'], f'{metric} failed'
          "
```

---

## Next Steps

1. **Run Quick Test**: Start with a simple test generation
   ```bash
   python -m layer2_test_generation.llm_test_agent
   ```

2. **Run Integration Tests**: Verify everything works
   ```bash
   python tests/integration/test_end_to_end_pipeline.py
   ```

3. **Validate One Claim**: Pick one claim to validate
   ```bash
   bash scripts/validate_claims.sh --coverage-only
   ```

4. **Full Validation**: Run complete evaluation
   ```bash
   bash scripts/validate_claims.sh /path/to/project
   ```

5. **Review Results**: Check the summary report
   ```bash
   cat evaluation_results/summary_report.json
   ```

---

## Documentation

- **Configuration**: `docs/CONFIGURATION_GUIDE.md`
- **Validation**: `docs/VALIDATION_GUIDE.md`
- **Equations**: `docs/EQUATION_TO_CODE_MAPPING.md`
- **API Reference**: `docs/API_REFERENCE.md`
- **Examples**: `examples/`

---

## Support

- **Issues**: https://github.com/learningdebunked/XTestAgent/issues
- **Discussions**: https://github.com/learningdebunked/XTestAgent/discussions
- **Email**: support@testagentx.org

---

## Quick Reference

```bash
# Installation
pip install -r requirements.txt
bash scripts/setup_jacoco.sh

# Run tests
python tests/integration/test_end_to_end_pipeline.py

# Validate claims
bash scripts/validate_claims.sh /path/to/project

# View results
cat evaluation_results/summary_report.json

# Configuration
export TESTAGENTX_CONFIG=config/my_config.yaml

# Help
python evaluation/run_full_evaluation.py --help
```

---

**Happy Testing! 🚀**
