#!/usr/bin/env bash
# Validation script for Parrot project
# Usage: bash validate_parrot.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TestAgentX - Parrot Project Validation${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configuration
PARROT_PATH="/Users/kapilsindhu/CascadeProjects/Parrot"
TESTAGENT_PATH="/Users/kapilsindhu/CascadeProjects/XTestAgent/TestAgent"
OUTPUT_DIR="${TESTAGENT_PATH}/evaluation_results/parrot_$(date +%Y%m%d_%H%M%S)"

# Check if Parrot exists
if [ ! -d "$PARROT_PATH" ]; then
    echo -e "${YELLOW}Parrot project not found. Cloning...${NC}"
    cd /Users/kapilsindhu/CascadeProjects
    git clone https://github.com/learningdebunked/Parrot.git
    echo -e "${GREEN}✓ Parrot cloned${NC}"
else
    echo -e "${GREEN}✓ Parrot project found${NC}"
fi

# Check Parrot structure
echo ""
echo -e "${YELLOW}Analyzing Parrot project structure...${NC}"
cd "$PARROT_PATH"

# Find Java files
JAVA_FILES=$(find . -name "*.java" -type f 2>/dev/null | wc -l)
echo -e "${GREEN}✓ Found ${JAVA_FILES} Java files${NC}"

# Check if it's a Maven project
if [ -f "pom.xml" ]; then
    echo -e "${GREEN}✓ Maven project detected${NC}"
    PROJECT_TYPE="maven"
elif [ -f "build.gradle" ]; then
    echo -e "${GREEN}✓ Gradle project detected${NC}"
    PROJECT_TYPE="gradle"
else
    echo -e "${YELLOW}⚠ No build file found (pom.xml or build.gradle)${NC}"
    PROJECT_TYPE="unknown"
fi

# List main source files
echo ""
echo -e "${YELLOW}Main source files:${NC}"
find ./src/main/java -name "*.java" -type f 2>/dev/null | head -10 || \
find . -name "*.java" -type f 2>/dev/null | head -10

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Validation Options${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. Generate tests for all classes"
echo "2. Generate tests for specific class"
echo "3. Run full validation (coverage, mutation, etc.)"
echo "4. Quick validation (basic checks)"
echo ""
read -p "Select option (1-4): " OPTION

case $OPTION in
    1)
        echo ""
        echo -e "${YELLOW}Generating tests for all classes...${NC}"
        cd "$TESTAGENT_PATH"
        
        # Find all Java source files
        for java_file in $(find "$PARROT_PATH/src/main/java" -name "*.java" -type f 2>/dev/null); do
            echo -e "${BLUE}Processing: $(basename $java_file)${NC}"
            
            # Generate output path
            rel_path=${java_file#$PARROT_PATH/src/main/java/}
            test_file="$PARROT_PATH/src/test/java/${rel_path%.java}Test.java"
            
            # Create test directory
            mkdir -p "$(dirname $test_file)"
            
            # Generate tests
            python3 -m layer2_test_generation.llm_test_agent \
                --source "$java_file" \
                --output "$test_file" \
                --num-tests 5 || echo -e "${RED}Failed for $(basename $java_file)${NC}"
        done
        
        echo -e "${GREEN}✓ Test generation complete${NC}"
        ;;
        
    2)
        echo ""
        echo -e "${YELLOW}Available classes:${NC}"
        find "$PARROT_PATH/src/main/java" -name "*.java" -type f 2>/dev/null | nl
        echo ""
        read -p "Enter the full path to the Java file: " JAVA_FILE
        
        if [ -f "$JAVA_FILE" ]; then
            echo -e "${YELLOW}Generating tests for $(basename $JAVA_FILE)...${NC}"
            cd "$TESTAGENT_PATH"
            
            # Generate output path
            rel_path=${JAVA_FILE#$PARROT_PATH/src/main/java/}
            test_file="$PARROT_PATH/src/test/java/${rel_path%.java}Test.java"
            mkdir -p "$(dirname $test_file)"
            
            python3 -m layer2_test_generation.llm_test_agent \
                --source "$JAVA_FILE" \
                --output "$test_file" \
                --num-tests 10
            
            echo -e "${GREEN}✓ Tests generated: $test_file${NC}"
        else
            echo -e "${RED}File not found: $JAVA_FILE${NC}"
            exit 1
        fi
        ;;
        
    3)
        echo ""
        echo -e "${YELLOW}Running full validation on Parrot...${NC}"
        cd "$TESTAGENT_PATH"
        
        python3 evaluation/run_full_evaluation.py \
            --project "$PARROT_PATH" \
            --output "$OUTPUT_DIR"
        
        echo ""
        echo -e "${GREEN}✓ Validation complete${NC}"
        echo -e "${BLUE}Results saved to: $OUTPUT_DIR${NC}"
        echo ""
        echo -e "${YELLOW}View results:${NC}"
        echo "  cat $OUTPUT_DIR/summary_report.json"
        ;;
        
    4)
        echo ""
        echo -e "${YELLOW}Running quick validation...${NC}"
        cd "$TESTAGENT_PATH"
        
        # Run simple checks
        python3 << EOF
import sys
from pathlib import Path

parrot_path = Path("$PARROT_PATH")

# Count files
java_files = list(parrot_path.rglob("*.java"))
test_files = list(parrot_path.rglob("*Test.java"))

print(f"📊 Parrot Project Statistics:")
print(f"  Total Java files: {len(java_files)}")
print(f"  Test files: {len(test_files)}")
print(f"  Source files: {len(java_files) - len(test_files)}")
print(f"  Test coverage: {len(test_files) / max(len(java_files) - len(test_files), 1) * 100:.1f}%")
print()
print("✅ Quick validation complete!")
EOF
        ;;
        
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Parrot Validation Complete${NC}"
echo -e "${BLUE}========================================${NC}"
