"""
Mutation Operators for TestAgentX

Implements various mutation operators for Java and Python code.
Mutation operators introduce small syntactic changes to test the quality
of test suites.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import re
import ast


class MutationOperatorType(Enum):
    """Types of mutation operators"""
    # Arithmetic operators
    AOR = "arithmetic_operator_replacement"  # +, -, *, /, %
    
    # Relational operators
    ROR = "relational_operator_replacement"  # <, >, <=, >=, ==, !=
    
    # Conditional operators
    COR = "conditional_operator_replacement"  # &&, ||
    
    # Logical operators
    LOR = "logical_operator_replacement"  # &, |, ^
    
    # Assignment operators
    ASOR = "assignment_operator_replacement"  # +=, -=, *=, /=
    
    # Unary operators
    UOI = "unary_operator_insertion"  # +, -, !
    UOD = "unary_operator_deletion"
    
    # Statement mutations
    SDL = "statement_deletion"
    
    # Constant mutations
    CRP = "constant_replacement"  # 0, 1, -1
    
    # Return mutations
    RVR = "return_value_replacement"
    
    # Null mutations
    NMR = "null_mutation_replacement"
    
    # Method call mutations
    MCR = "method_call_replacement"
    
    # Boundary mutations
    BVA = "boundary_value_adjustment"  # i < n -> i <= n


@dataclass
class MutationOperator:
    """Represents a mutation operator"""
    operator_type: MutationOperatorType
    name: str
    description: str
    language: str  # 'java' or 'python'
    pattern: str  # Regex pattern to match
    replacement_fn: Any  # Function to generate replacement
    
    def apply(self, code: str, line_number: int) -> Optional[str]:
        """Apply mutation to code at specific line"""
        lines = code.split('\n')
        if line_number < 0 or line_number >= len(lines):
            return None
        
        original_line = lines[line_number]
        mutated_line = self.replacement_fn(original_line)
        
        if mutated_line and mutated_line != original_line:
            lines[line_number] = mutated_line
            return '\n'.join(lines)
        
        return None


# Java Mutation Operators

def java_aor_replacement(line: str) -> str:
    """Arithmetic Operator Replacement for Java"""
    replacements = {
        r'\+': '-',
        r'-': '+',
        r'\*': '/',
        r'/': '*',
        r'%': '*'
    }
    
    for pattern, replacement in replacements.items():
        if re.search(pattern, line):
            return re.sub(pattern, replacement, line, count=1)
    return line


def java_ror_replacement(line: str) -> str:
    """Relational Operator Replacement for Java"""
    replacements = {
        r'<(?!=)': '<=',
        r'<=': '<',
        r'>(?!=)': '>=',
        r'>=': '>',
        r'==': '!=',
        r'!=': '=='
    }
    
    for pattern, replacement in replacements.items():
        if re.search(pattern, line):
            return re.sub(pattern, replacement, line, count=1)
    return line


def java_cor_replacement(line: str) -> str:
    """Conditional Operator Replacement for Java"""
    if '&&' in line:
        return line.replace('&&', '||', 1)
    elif '||' in line:
        return line.replace('||', '&&', 1)
    return line


def java_uoi_insertion(line: str) -> str:
    """Unary Operator Insertion for Java"""
    # Insert negation
    match = re.search(r'if\s*\(\s*(\w+)\s*\)', line)
    if match:
        var = match.group(1)
        return line.replace(f'({var})', f'(!{var})', 1)
    return line


def java_sdl_deletion(line: str) -> str:
    """Statement Deletion for Java"""
    # Comment out the line
    if line.strip() and not line.strip().startswith('//'):
        return '// MUTANT: ' + line
    return line


def java_crp_replacement(line: str) -> str:
    """Constant Replacement for Java"""
    # Replace numeric constants
    if re.search(r'\b0\b', line):
        return re.sub(r'\b0\b', '1', line, count=1)
    elif re.search(r'\b1\b', line):
        return re.sub(r'\b1\b', '0', line, count=1)
    return line


def java_rvr_replacement(line: str) -> str:
    """Return Value Replacement for Java"""
    if 'return true' in line:
        return line.replace('return true', 'return false', 1)
    elif 'return false' in line:
        return line.replace('return false', 'return true', 1)
    elif re.search(r'return\s+\d+', line):
        return re.sub(r'return\s+(\d+)', r'return 0', line, count=1)
    return line


def java_nmr_replacement(line: str) -> str:
    """Null Mutation Replacement for Java"""
    # Replace object returns with null
    if re.search(r'return\s+new\s+', line):
        return re.sub(r'return\s+new\s+.*?;', 'return null;', line, count=1)
    return line


def java_bva_adjustment(line: str) -> str:
    """Boundary Value Adjustment for Java"""
    if '<' in line and '<=' not in line:
        return line.replace('<', '<=', 1)
    elif '<=' in line:
        return line.replace('<=', '<', 1)
    elif '>' in line and '>=' not in line:
        return line.replace('>', '>=', 1)
    elif '>=' in line:
        return line.replace('>=', '>', 1)
    return line


# Python Mutation Operators

def python_aor_replacement(line: str) -> str:
    """Arithmetic Operator Replacement for Python"""
    replacements = {
        r'\+': '-',
        r'-': '+',
        r'\*': '/',
        r'/': '*',
        r'%': '*'
    }
    
    for pattern, replacement in replacements.items():
        if re.search(pattern, line):
            return re.sub(pattern, replacement, line, count=1)
    return line


def python_ror_replacement(line: str) -> str:
    """Relational Operator Replacement for Python"""
    replacements = {
        r'<(?!=)': '<=',
        r'<=': '<',
        r'>(?!=)': '>=',
        r'>=': '>',
        r'==': '!=',
        r'!=': '=='
    }
    
    for pattern, replacement in replacements.items():
        if re.search(pattern, line):
            return re.sub(pattern, replacement, line, count=1)
    return line


def python_cor_replacement(line: str) -> str:
    """Conditional Operator Replacement for Python"""
    if ' and ' in line:
        return line.replace(' and ', ' or ', 1)
    elif ' or ' in line:
        return line.replace(' or ', ' and ', 1)
    return line


def python_uoi_insertion(line: str) -> str:
    """Unary Operator Insertion for Python"""
    # Insert not
    match = re.search(r'if\s+(\w+):', line)
    if match:
        var = match.group(1)
        return line.replace(f'if {var}:', f'if not {var}:', 1)
    return line


def python_sdl_deletion(line: str) -> str:
    """Statement Deletion for Python"""
    # Comment out the line
    if line.strip() and not line.strip().startswith('#'):
        return '# MUTANT: ' + line
    return line


def python_crp_replacement(line: str) -> str:
    """Constant Replacement for Python"""
    if re.search(r'\b0\b', line):
        return re.sub(r'\b0\b', '1', line, count=1)
    elif re.search(r'\b1\b', line):
        return re.sub(r'\b1\b', '0', line, count=1)
    return line


def python_rvr_replacement(line: str) -> str:
    """Return Value Replacement for Python"""
    if 'return True' in line:
        return line.replace('return True', 'return False', 1)
    elif 'return False' in line:
        return line.replace('return False', 'return True', 1)
    elif re.search(r'return\s+\d+', line):
        return re.sub(r'return\s+(\d+)', r'return 0', line, count=1)
    return line


def python_nmr_replacement(line: str) -> str:
    """Null Mutation Replacement for Python"""
    # Replace object returns with None
    if re.search(r'return\s+\w+\(', line):
        return re.sub(r'return\s+.*', 'return None', line, count=1)
    return line


def python_bva_adjustment(line: str) -> str:
    """Boundary Value Adjustment for Python"""
    if '<' in line and '<=' not in line:
        return line.replace('<', '<=', 1)
    elif '<=' in line:
        return line.replace('<=', '<', 1)
    elif '>' in line and '>=' not in line:
        return line.replace('>', '>=', 1)
    elif '>=' in line:
        return line.replace('>=', '>', 1)
    return line


# Predefined Mutation Operators

JAVA_MUTATION_OPERATORS = [
    MutationOperator(
        operator_type=MutationOperatorType.AOR,
        name="Arithmetic Operator Replacement",
        description="Replace +, -, *, /, % with each other",
        language="java",
        pattern=r'[\+\-\*/%]',
        replacement_fn=java_aor_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.ROR,
        name="Relational Operator Replacement",
        description="Replace <, >, <=, >=, ==, != with each other",
        language="java",
        pattern=r'[<>]=?|[!=]=',
        replacement_fn=java_ror_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.COR,
        name="Conditional Operator Replacement",
        description="Replace && with || and vice versa",
        language="java",
        pattern=r'&&|\|\|',
        replacement_fn=java_cor_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.UOI,
        name="Unary Operator Insertion",
        description="Insert negation operator",
        language="java",
        pattern=r'if\s*\(',
        replacement_fn=java_uoi_insertion
    ),
    MutationOperator(
        operator_type=MutationOperatorType.SDL,
        name="Statement Deletion",
        description="Delete statements",
        language="java",
        pattern=r'.*',
        replacement_fn=java_sdl_deletion
    ),
    MutationOperator(
        operator_type=MutationOperatorType.CRP,
        name="Constant Replacement",
        description="Replace constants 0, 1",
        language="java",
        pattern=r'\b[01]\b',
        replacement_fn=java_crp_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.RVR,
        name="Return Value Replacement",
        description="Replace return values",
        language="java",
        pattern=r'return\s+',
        replacement_fn=java_rvr_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.NMR,
        name="Null Mutation Replacement",
        description="Replace object returns with null",
        language="java",
        pattern=r'return\s+new',
        replacement_fn=java_nmr_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.BVA,
        name="Boundary Value Adjustment",
        description="Adjust boundary conditions",
        language="java",
        pattern=r'[<>]=?',
        replacement_fn=java_bva_adjustment
    )
]

PYTHON_MUTATION_OPERATORS = [
    MutationOperator(
        operator_type=MutationOperatorType.AOR,
        name="Arithmetic Operator Replacement",
        description="Replace +, -, *, /, % with each other",
        language="python",
        pattern=r'[\+\-\*/%]',
        replacement_fn=python_aor_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.ROR,
        name="Relational Operator Replacement",
        description="Replace <, >, <=, >=, ==, != with each other",
        language="python",
        pattern=r'[<>]=?|[!=]=',
        replacement_fn=python_ror_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.COR,
        name="Conditional Operator Replacement",
        description="Replace and with or and vice versa",
        language="python",
        pattern=r'\band\b|\bor\b',
        replacement_fn=python_cor_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.UOI,
        name="Unary Operator Insertion",
        description="Insert not operator",
        language="python",
        pattern=r'if\s+\w+:',
        replacement_fn=python_uoi_insertion
    ),
    MutationOperator(
        operator_type=MutationOperatorType.SDL,
        name="Statement Deletion",
        description="Delete statements",
        language="python",
        pattern=r'.*',
        replacement_fn=python_sdl_deletion
    ),
    MutationOperator(
        operator_type=MutationOperatorType.CRP,
        name="Constant Replacement",
        description="Replace constants 0, 1",
        language="python",
        pattern=r'\b[01]\b',
        replacement_fn=python_crp_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.RVR,
        name="Return Value Replacement",
        description="Replace return values",
        language="python",
        pattern=r'return\s+',
        replacement_fn=python_rvr_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.NMR,
        name="Null Mutation Replacement",
        description="Replace object returns with None",
        language="python",
        pattern=r'return\s+\w+\(',
        replacement_fn=python_nmr_replacement
    ),
    MutationOperator(
        operator_type=MutationOperatorType.BVA,
        name="Boundary Value Adjustment",
        description="Adjust boundary conditions",
        language="python",
        pattern=r'[<>]=?',
        replacement_fn=python_bva_adjustment
    )
]


def get_operators_for_language(language: str) -> List[MutationOperator]:
    """Get mutation operators for a specific language"""
    if language.lower() == 'java':
        return JAVA_MUTATION_OPERATORS
    elif language.lower() == 'python':
        return PYTHON_MUTATION_OPERATORS
    else:
        return []


def get_operator_by_type(operator_type: MutationOperatorType, 
                        language: str) -> Optional[MutationOperator]:
    """Get a specific mutation operator by type and language"""
    operators = get_operators_for_language(language)
    for op in operators:
        if op.operator_type == operator_type:
            return op
    return None
