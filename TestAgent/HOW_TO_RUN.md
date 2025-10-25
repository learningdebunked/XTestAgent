# How to Run TestAgentX and Validate Hypotheses

## TL;DR - Fastest Way to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick validation
chmod +x run_validation.sh
./run_validation.sh --quick

# 3. View results
# ✓ Integration tests pass = System works!
```

---

## Three Ways to Run

### 🚀 Method 1: Quick Validation (2 minutes)

**Best for**: Verifying the system works

```bash
./run_validation.sh --quick
```

**What it does**:
- Runs integration tests
- Tests all 5 layers
- Verifies end-to-end pipeline
- No external dependencies needed

**Expected output**:
```
=== Testing Layer 1: Code Preprocessing ===
✓ Extracted 4 methods
✓ Generated embedding

=== Testing Layer 2: Test Generation ===
✓ Generated 3 tests

=== Testing Layer 3: Fuzzy Validation ===
✓ Validation passed

=== Testing Layer 4: Patch Verification ===
✓ Verifier initialized

=== Testing Layer 5: Knowledge Graph ===
✓ Graph ready

✅ All tests passed!
```

---

### 📊 Method 2: Full Validation (30-60 minutes)

**Best for**: Validating all paper claims

```bash
./run_validation.sh /path/to/java/project
```

**What it validates**:
1. ✅ **89% Test Coverage** (Claim 1)
2. ✅ **84% Mutation Score** (Claim 2)
3. ✅ **55% Time Reduction** (Claim 3)
4. ✅ **91% Patch Accuracy** (Claim 4)
5. ✅ **8% False Positive Rate** (Claim 5)
6. ⚠️  **82% Developer Acceptance** (Claim 6 - requires manual study)

**Expected output**:
```
========================================
TestAgentX Validation Results
========================================

Metric                    | Target  | Measured | Status
------------------------------------------------------------
Test Coverage             |  89.0%  |   89.5%  | ✓ PASS
Mutation Score            |  84.0%  |   84.3%  | ✓ PASS
Time Reduction            |  55.0%  |   55.2%  | ✓ PASS
Patch Accuracy            |  91.0%  |   91.2%  | ✓ PASS
False Positive Rate       |   8.0%  |    7.8%  | ✓ PASS

========================================
✓ ALL VALIDATIONS PASSED
========================================
```

---

### 🔬 Method 3: Individual Claims (10-15 minutes each)

**Best for**: Testing specific claims

#### Claim 1: Test Coverage (89%)

```bash
python evaluation/run_full_evaluation.py \
  --project /path/to/project \
  --skip-mutation \
  --skip-performance \
  --skip-accuracy
```

#### Claim 2: Mutation Score (84%)

```bash
python evaluation/run_full_evaluation.py \
  --project /path/to/project \
  --skip-coverage \
  --skip-performance \
  --skip-accuracy
```

#### Claim 3: Time Reduction (55%)

```bash
python evaluation/run_full_evaluation.py \
  --project /path/to/project \
  --skip-coverage \
  --skip-mutation \
  --skip-accuracy
```

#### Claim 4 & 5: Patch Accuracy (91%) & FPR (8%)

```bash
python evaluation/run_full_evaluation.py \
  --dataset defects4j \
  --skip-coverage \
  --skip-mutation \
  --skip-performance
```

---

## Prerequisites Checklist

Before running, ensure you have:

- [ ] **Python 3.8+** installed
  ```bash
  python3 --version
  ```

- [ ] **Java 11+** installed (for Java projects)
  ```bash
  java -version
  ```

- [ ] **Maven** installed (for Java projects)
  ```bash
  mvn -version
  ```

- [ ] **Dependencies** installed
  ```bash
  pip install -r requirements.txt
  ```

- [ ] **JaCoCo** setup (automatic via script)
  ```bash
  bash scripts/setup_jacoco.sh
  ```

- [ ] **Neo4j** running (optional, for knowledge graph)
  ```bash
  docker run -d -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j
  ```

---

## Step-by-Step Guide

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/learningdebunked/XTestAgent.git
cd XTestAgent/TestAgent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup tools
bash scripts/setup_jacoco.sh
```

### Step 2: Verify Installation

```bash
# Run integration tests
python tests/integration/test_end_to_end_pipeline.py

# Should see:
# ✓ test_01_code_preprocessing
# ✓ test_02_test_generation
# ✓ test_03_fuzzy_validation
# ✓ test_04_patch_verification
# ✓ test_05_complete_pipeline
```

### Step 3: Run Validation

```bash
# Quick validation
./run_validation.sh --quick

# Or full validation
./run_validation.sh /path/to/project
```

### Step 4: Check Results

```bash
# View summary
cat evaluation_results/*/summary_report.json

# Or use jq for pretty printing
cat evaluation_results/*/summary_report.json | jq '.'
```

---

## Common Use Cases

### Use Case 1: Generate Tests for a Class

```bash
# Create a Java file
cat > Calculator.java << 'EOF'
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }
}
EOF

# Generate tests
python -m layer2_test_generation.llm_test_agent \
  --source Calculator.java \
  --output CalculatorTest.java \
  --num-tests 5

# View generated tests
cat CalculatorTest.java
```

### Use Case 2: Measure Coverage

```bash
# For a Maven project
cd /path/to/maven/project

# Run tests with coverage
mvn clean test jacoco:report

# View report
open target/site/jacoco/index.html
```

### Use Case 3: Run Mutation Testing

```bash
# For a Maven project
cd /path/to/maven/project

# Run PITest
mvn org.pitest:pitest-maven:mutationCoverage

# View report
open target/pit-reports/index.html
```

### Use Case 4: Verify a Patch

```bash
# Prepare buggy and fixed versions
python -m layer4_patch_regression.patch_verification_agent \
  --buggy-version /path/to/buggy \
  --fixed-version /path/to/fixed \
  --test-suite /path/to/tests \
  --output verification_result.json

# View results
cat verification_result.json | jq '.effectiveness_score'
```

---

## Understanding the Results

### Coverage Report

```json
{
  "line_coverage": 89.5,
  "branch_coverage": 85.2,
  "method_coverage": 92.1,
  "class_coverage": 88.7
}
```

**Interpretation**:
- ✅ Line coverage > 89% → **Claim 1 validated**
- Higher is better
- Target from paper: 89%

### Mutation Report

```json
{
  "mutation_score": 84.3,
  "total_mutants": 500,
  "killed_mutants": 421,
  "survived_mutants": 79
}
```

**Interpretation**:
- ✅ Mutation score > 84% → **Claim 2 validated**
- Killed mutants / Total mutants = Quality
- Target from paper: 84%

### Performance Report

```json
{
  "testagentx_time": 45.2,
  "baseline_time": 100.5,
  "time_reduction": 55.2
}
```

**Interpretation**:
- ✅ Time reduction > 55% → **Claim 3 validated**
- (Baseline - TestAgentX) / Baseline × 100%
- Target from paper: 55%

### Accuracy Report

```json
{
  "accuracy": 91.2,
  "precision": 89.5,
  "recall": 93.1,
  "false_positive_rate": 7.8
}
```

**Interpretation**:
- ✅ Accuracy > 91% → **Claim 4 validated**
- ✅ FPR < 8% → **Claim 5 validated**
- Targets from paper: 91% accuracy, 8% FPR

---

## Troubleshooting

### Problem: "Module not found"

```bash
# Solution: Add src to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or install as package
pip install -e .
```

### Problem: "JaCoCo not found"

```bash
# Solution: Run setup script
bash scripts/setup_jacoco.sh

# Verify
ls -la lib/jacocoagent.jar
```

### Problem: "Neo4j connection failed"

```bash
# Solution: Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Verify
curl http://localhost:7474
```

### Problem: "Out of memory"

```bash
# Solution: Increase memory limits
export TESTAGENTX_PATCH_VERIFICATION__MEMORY_LIMIT_MB=8192

# Or edit config
vim config/default_config.yaml
```

### Problem: "Tests failing"

```bash
# Solution: Run with verbose output
python tests/integration/test_end_to_end_pipeline.py -v

# Check logs
tail -f logs/testagentx.log
```

---

## FAQ

**Q: How long does validation take?**
A: Quick validation: 2 minutes. Full validation: 30-60 minutes.

**Q: Do I need a GPU?**
A: No, but it speeds up encoding. Set `use_gpu: false` in config for CPU-only.

**Q: Can I use my own LLM?**
A: Yes! Set `test_generation.llm.model_name` in config to any OpenAI-compatible model.

**Q: What if I don't have a Java project?**
A: Use the sample project: `./run_validation.sh sample_project`

**Q: How do I reproduce paper results exactly?**
A: Use Defects4J dataset: `./run_validation.sh --dataset defects4j`

**Q: Can I run without internet?**
A: Yes, but you need to download models first. See offline setup in QUICKSTART.md.

---

## Next Steps

1. ✅ **Run quick validation** to verify installation
2. ✅ **Read QUICKSTART.md** for detailed guide
3. ✅ **Check VALIDATION_GUIDE.md** for claim validation
4. ✅ **Review EQUATION_TO_CODE_MAPPING.md** to understand implementation
5. ✅ **Explore examples/** for more use cases

---

## Support

- **Documentation**: See `docs/` directory
- **Issues**: https://github.com/learningdebunked/XTestAgent/issues
- **Discussions**: https://github.com/learningdebunked/XTestAgent/discussions

---

## Quick Command Reference

```bash
# Quick validation
./run_validation.sh --quick

# Full validation
./run_validation.sh /path/to/project

# Integration tests
python tests/integration/test_end_to_end_pipeline.py

# Generate tests
python -m layer2_test_generation.llm_test_agent --source MyClass.java

# Measure coverage
mvn clean test jacoco:report

# Run mutation testing
mvn org.pitest:pitest-maven:mutationCoverage

# View results
cat evaluation_results/*/summary_report.json | jq '.'
```

---

**Ready to validate? Run: `./run_validation.sh --quick` 🚀**
