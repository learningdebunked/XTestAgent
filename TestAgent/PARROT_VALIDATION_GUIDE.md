# Validating TestAgentX with Parrot Project

This guide shows how to use your Parrot project (https://github.com/learningdebunked/Parrot) to validate the TestAgentX framework.

## Quick Start

```bash
# Make the script executable
chmod +x validate_parrot.sh

# Run the validation script
bash validate_parrot.sh
```

The script will guide you through 4 options:
1. Generate tests for all classes
2. Generate tests for specific class
3. Run full validation
4. Quick validation

---

## Manual Validation Steps

### Step 1: Clone Parrot (if needed)

```bash
cd /Users/kapilsindhu/CascadeProjects

# Clone if not already present
git clone https://github.com/learningdebunked/Parrot.git
cd Parrot

# Check the structure
ls -la
tree -L 3  # if tree is installed
```

### Step 2: Analyze Parrot Project

```bash
# Find all Java files
find . -name "*.java" -type f

# Check if it's Maven or Gradle
ls pom.xml build.gradle 2>/dev/null

# View project structure
find ./src -type d 2>/dev/null
```

### Step 3: Generate Tests for One Class

```bash
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent

# Set API key (if using LLM)
export OPENAI_API_KEY=your_key_here

# Example: Generate tests for a specific class
# Replace with actual class from Parrot
python3 -m layer2_test_generation.llm_test_agent \
  --source /Users/kapilsindhu/CascadeProjects/Parrot/src/main/java/com/example/YourClass.java \
  --output /Users/kapilsindhu/CascadeProjects/Parrot/src/test/java/com/example/YourClassTest.java \
  --num-tests 10 \
  --model gpt-4
```

### Step 4: Run Tests

```bash
cd /Users/kapilsindhu/CascadeProjects/Parrot

# If Maven project
mvn clean test

# If Gradle project
./gradlew test

# View test results
cat target/surefire-reports/*.txt  # Maven
cat build/test-results/test/*.xml  # Gradle
```

### Step 5: Measure Coverage

```bash
# Maven with JaCoCo
mvn clean test jacoco:report
open target/site/jacoco/index.html

# Gradle with JaCoCo
./gradlew test jacocoTestReport
open build/reports/jacoco/test/html/index.html
```

### Step 6: Run Full Validation

```bash
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent

python3 evaluation/run_full_evaluation.py \
  --project /Users/kapilsindhu/CascadeProjects/Parrot \
  --output evaluation_results/parrot_validation/

# View results
cat evaluation_results/parrot_validation/summary_report.json | jq '.'
```

---

## Validation Scenarios

### Scenario 1: Test Coverage Validation

**Goal**: Verify TestAgentX achieves 89% coverage on Parrot

```bash
# Generate tests for all Parrot classes
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent

# Find all Java source files
for file in $(find /Users/kapilsindhu/CascadeProjects/Parrot/src/main/java -name "*.java"); do
    echo "Processing: $file"
    
    # Extract package and class name
    rel_path=${file#/Users/kapilsindhu/CascadeProjects/Parrot/src/main/java/}
    test_file="/Users/kapilsindhu/CascadeProjects/Parrot/src/test/java/${rel_path%.java}Test.java"
    
    # Create directory
    mkdir -p "$(dirname $test_file)"
    
    # Generate tests
    python3 -m layer2_test_generation.llm_test_agent \
        --source "$file" \
        --output "$test_file" \
        --num-tests 5
done

# Measure coverage
cd /Users/kapilsindhu/CascadeProjects/Parrot
mvn clean test jacoco:report

# Check if coverage >= 89%
python3 << 'EOF'
import xml.etree.ElementTree as ET

tree = ET.parse('target/site/jacoco/jacoco.xml')
root = tree.getroot()

for counter in root.findall('.//counter[@type="LINE"]'):
    covered = int(counter.get('covered'))
    missed = int(counter.get('missed'))
    total = covered + missed
    coverage = (covered / total * 100) if total > 0 else 0
    
    print(f"Line Coverage: {coverage:.1f}%")
    if coverage >= 89.0:
        print("✅ Meets paper claim (89%)")
    else:
        print(f"❌ Below target (need {89.0 - coverage:.1f}% more)")
EOF
```

### Scenario 2: Mutation Testing Validation

**Goal**: Verify TestAgentX achieves 84% mutation score on Parrot

```bash
# Add PITest to Parrot's pom.xml (if Maven)
cd /Users/kapilsindhu/CascadeProjects/Parrot

# Run mutation testing
mvn org.pitest:pitest-maven:mutationCoverage

# View results
open target/pit-reports/index.html

# Check mutation score
grep -A 5 "Mutation Coverage" target/pit-reports/index.html
```

### Scenario 3: Time Comparison

**Goal**: Verify TestAgentX is 55% faster than baseline

```bash
# Time TestAgentX
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent
time python3 -m layer2_test_generation.llm_test_agent \
    --source /Users/kapilsindhu/CascadeProjects/Parrot/src/main/java/com/example/YourClass.java \
    --num-tests 10

# Time EvoSuite (baseline)
time java -jar evosuite.jar \
    -class com.example.YourClass \
    -projectCP /Users/kapilsindhu/CascadeProjects/Parrot/target/classes

# Compare times
```

---

## Expected Results for Parrot

Based on the paper's claims, you should see:

| Metric | Target | Expected on Parrot |
|--------|--------|--------------------|
| Test Coverage | 89% | 85-92% |
| Mutation Score | 84% | 80-88% |
| Time Reduction | 55% | 50-60% |
| Patch Accuracy | 91% | 88-94% |
| False Positive Rate | 8% | 5-10% |

---

## Troubleshooting Parrot-Specific Issues

### Issue: Parrot uses specific Java version

```bash
# Check Parrot's Java requirements
cat /Users/kapilsindhu/CascadeProjects/Parrot/pom.xml | grep -A 5 "maven.compiler"

# Set Java version
export JAVA_HOME=/path/to/jdk-11
```

### Issue: Parrot has dependencies

```bash
# Install Parrot dependencies first
cd /Users/kapilsindhu/CascadeProjects/Parrot
mvn clean install -DskipTests

# Then generate tests
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent
# ... continue with test generation
```

### Issue: Parrot uses custom build system

```bash
# Check build system
cd /Users/kapilsindhu/CascadeProjects/Parrot
ls pom.xml build.gradle build.xml Makefile

# Adapt commands accordingly
```

---

## Analyzing Parrot Results

### View Coverage Report

```bash
# After running tests with coverage
cd /Users/kapilsindhu/CascadeProjects/Parrot

# Maven
open target/site/jacoco/index.html

# Or view in terminal
cat target/site/jacoco/index.html | grep -A 2 "Total"
```

### View Mutation Report

```bash
# After running PITest
cd /Users/kapilsindhu/CascadeProjects/Parrot
open target/pit-reports/index.html

# Or extract key metrics
python3 << 'EOF'
import xml.etree.ElementTree as ET
tree = ET.parse('target/pit-reports/mutations.xml')
root = tree.getroot()

total = len(root.findall('.//mutation'))
killed = len(root.findall('.//mutation[@status="KILLED"]'))
score = (killed / total * 100) if total > 0 else 0

print(f"Mutation Score: {score:.1f}%")
print(f"Mutants Killed: {killed}/{total}")
EOF
```

### Compare with Baseline

```bash
# Generate comparison report
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent

python3 << 'EOF'
import json

# Load TestAgentX results
with open('evaluation_results/parrot_validation/summary_report.json') as f:
    results = json.load(f)

print("="*60)
print("TestAgentX Performance on Parrot")
print("="*60)

for metric, values in results.get('comparison', {}).items():
    print(f"\n{metric.replace('_', ' ').title()}:")
    print(f"  Paper Claim: {values['paper_claim']:.1f}%")
    print(f"  Measured:    {values['measured']:.1f}%")
    print(f"  Difference:  {values['difference']:+.1f}%")
    print(f"  Status:      {'✅ PASS' if values['meets_claim'] else '❌ FAIL'}")

print("\n" + "="*60)
EOF
```

---

## Creating a Report

### Generate Markdown Report

```bash
cd /Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent

python3 << 'EOF'
import json
from datetime import datetime

# Load results
with open('evaluation_results/parrot_validation/summary_report.json') as f:
    results = json.load(f)

# Generate report
report = f"""# TestAgentX Validation Report - Parrot Project

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Project**: Parrot (https://github.com/learningdebunked/Parrot)
**Framework**: TestAgentX

## Summary

| Metric | Target | Measured | Status |
|--------|--------|----------|--------|
"""

for metric, values in results.get('comparison', {}).items():
    name = metric.replace('_', ' ').title()
    target = values['paper_claim']
    measured = values['measured']
    status = '✅ PASS' if values['meets_claim'] else '❌ FAIL'
    report += f"| {name} | {target:.1f}% | {measured:.1f}% | {status} |\n"

report += """
## Conclusion

"""

passed = sum(1 for v in results.get('comparison', {}).values() if v['meets_claim'])
total = len(results.get('comparison', {}))

if passed == total:
    report += f"✅ **All {total} metrics passed validation on Parrot project.**\n"
else:
    report += f"⚠️ **{passed}/{total} metrics passed validation on Parrot project.**\n"

# Save report
with open('evaluation_results/parrot_validation/REPORT.md', 'w') as f:
    f.write(report)

print("Report generated: evaluation_results/parrot_validation/REPORT.md")
print(report)
EOF
```

---

## Next Steps

After validating with Parrot:

1. **Document Results**: Save the validation report
2. **Compare Projects**: Try validation on other projects
3. **Tune Parameters**: Adjust config based on Parrot results
4. **Contribute**: Share findings with the community

---

## Quick Commands Reference

```bash
# Clone Parrot
git clone https://github.com/learningdebunked/Parrot.git

# Run automated validation
bash validate_parrot.sh

# Generate tests for one class
python3 -m layer2_test_generation.llm_test_agent --source Parrot/src/.../Class.java

# Run full validation
python3 evaluation/run_full_evaluation.py --project Parrot/

# View results
cat evaluation_results/parrot_validation/summary_report.json | jq '.'
```

---

**Ready to validate with Parrot?** Run:
```bash
bash validate_parrot.sh
```

🚀 **Good luck with your validation!**
