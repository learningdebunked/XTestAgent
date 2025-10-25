#!/usr/bin/env bash
# TestAgentX - Quick Validation Script
# Usage: bash run_validation.sh [project_path]
# Or: ./run_validation.sh [project_path] (after chmod +x)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="evaluation_results/$(date +%Y%m%d_%H%M%S)"

# Check for --quick flag
if [ "$1" == "--quick" ]; then
    QUICK_MODE=true
    PROJECT_PATH="sample_project"
else
    QUICK_MODE=false
    PROJECT_PATH="${1:-sample_project}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TestAgentX Validation Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
if [ "$QUICK_MODE" = true ]; then
    echo -e "Mode:    ${GREEN}Quick Validation (Integration Tests)${NC}"
else
    echo -e "Project: ${GREEN}${PROJECT_PATH}${NC}"
    echo -e "Output:  ${GREEN}${OUTPUT_DIR}${NC}"
fi
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python 3 found${NC}"

# Check Java
if ! command -v java &> /dev/null; then
    echo -e "${YELLOW}⚠ Java not found (required for Java projects)${NC}"
else
    echo -e "${GREEN}✓ Java found${NC}"
fi

# Check Maven
if ! command -v mvn &> /dev/null; then
    echo -e "${YELLOW}⚠ Maven not found (required for Java projects)${NC}"
else
    echo -e "${GREEN}✓ Maven found${NC}"
fi

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found${NC}"
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check dependencies
echo ""
echo -e "${YELLOW}Checking dependencies...${NC}"

if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import torch" &> /dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q -r requirements.txt
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check JaCoCo
echo ""
echo -e "${YELLOW}Checking JaCoCo...${NC}"

if [ ! -f "lib/jacocoagent.jar" ]; then
    echo -e "${YELLOW}Setting up JaCoCo...${NC}"
    bash scripts/setup_jacoco.sh
fi
echo -e "${GREEN}✓ JaCoCo ready${NC}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run validation
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Option 1: Quick validation (integration tests only)
if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}Running quick validation (simple tests)...${NC}"
    echo ""
    
    # Run simple test (safe on all platforms)
    python3 tests/simple_test.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}✓ Quick Validation PASSED${NC}"
        echo -e "${GREEN}========================================${NC}"
    else
        echo ""
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}✗ Quick Validation FAILED${NC}"
        echo -e "${RED}========================================${NC}"
        exit 1
    fi
    exit 0
fi

# Option 2: Full validation
echo -e "${YELLOW}Running full validation...${NC}"
echo ""

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/src"

# Run evaluation
python3 evaluation/run_full_evaluation.py \
    --project "${PROJECT_PATH}" \
    --output "${OUTPUT_DIR}" \
    2>&1 | tee "${OUTPUT_DIR}/validation.log"

# Check if summary report was generated
if [ -f "${OUTPUT_DIR}/summary_report.json" ]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Validation Results${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    # Parse and display results
    python3 << EOF
import json
import sys

try:
    with open('${OUTPUT_DIR}/summary_report.json', 'r') as f:
        data = json.load(f)
    
    comparison = data.get('comparison', {})
    all_passed = True
    
    print("Metric                    | Target  | Measured | Status")
    print("-" * 60)
    
    for metric, values in comparison.items():
        name = metric.replace('_', ' ').title()
        target = values['paper_claim']
        measured = values['measured']
        passed = values['meets_claim']
        status = '✓ PASS' if passed else '✗ FAIL'
        
        print(f'{name:<25} | {target:>6.1f}% | {measured:>7.1f}% | {status}')
        
        if not passed:
            all_passed = False
    
    print("")
    
    if all_passed:
        print('\033[0;32m' + '='*60 + '\033[0m')
        print('\033[0;32m' + '✓ ALL VALIDATIONS PASSED' + '\033[0m')
        print('\033[0;32m' + '='*60 + '\033[0m')
        sys.exit(0)
    else:
        print('\033[0;31m' + '='*60 + '\033[0m')
        print('\033[0;31m' + '✗ SOME VALIDATIONS FAILED' + '\033[0m')
        print('\033[0;31m' + '='*60 + '\033[0m')
        sys.exit(1)

except Exception as e:
    print(f'\033[0;31mError parsing results: {e}\033[0m')
    sys.exit(1)
EOF
    
    EXIT_CODE=$?
    
    echo ""
    echo -e "Full report: ${GREEN}${OUTPUT_DIR}/summary_report.json${NC}"
    echo -e "Log file:    ${GREEN}${OUTPUT_DIR}/validation.log${NC}"
    
    exit $EXIT_CODE
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Validation Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "Check log file: ${OUTPUT_DIR}/validation.log"
    exit 1
fi
