"""
Implements Section 3.3.1 from the paper: Bug Instance Ingestion
Loads bug instances from Defects4J dataset.
"""

import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
import yaml

@dataclass
class BugInstance:
    """Represents a bug from Defects4J"""
    project_id: str
    bug_id: int
    buggy_version_path: Path
    fixed_version_path: Path
    original_test_suite: List[str]
    commit_message: str
    modified_classes: List[str]
    trigger_tests: List[str]

class Defects4JLoader:
    """
    Loads and manages bug instances from Defects4J dataset.
    Corresponds to P_b (buggy) and P_f (fixed) from paper Section 3.3.1.
    """
    
    def __init__(self, defects4j_path: str = "/defects4j"):
        self.defects4j_path = Path(defects4j_path)
        self.workspace = Path("./data/defects4j/workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
    
    def load_bug(self, project: str, bug_id: int) -> BugInstance:
        """
        Load a specific bug instance from Defects4J.
        
        Args:
            project: Project name (e.g., 'Lang', 'Chart')
            bug_id: Bug number
            
        Returns:
            BugInstance with buggy (P_b) and fixed (P_f) versions
        """
        # Checkout buggy version
        buggy_dir = self.workspace / f"{project}_{bug_id}_buggy"
        self._checkout_version(project, bug_id, "buggy", buggy_dir)
        
        # Checkout fixed version
        fixed_dir = self.workspace / f"{project}_{bug_id}_fixed"
        self._checkout_version(project, bug_id, "fixed", fixed_dir)
        
        # Get bug metadata
        metadata = self._get_bug_metadata(project, bug_id)
        
        # Get original test suite (T_orig from paper)
        test_suite = self._extract_test_suite(buggy_dir)
        
        return BugInstance(
            project_id=project,
            bug_id=bug_id,
            buggy_version_path=buggy_dir,
            fixed_version_path=fixed_dir,
            original_test_suite=test_suite,
            commit_message=metadata.get('commit_message', ''),
            modified_classes=metadata.get('modified_classes', []),
            trigger_tests=metadata.get('trigger_tests', [])
        )
    
    def _checkout_version(self, project: str, bug_id: int, 
                         version: str, target_dir: Path) -> None:
        """Checkout specific version using Defects4J"""
        version_flag = "-b" if version == "buggy" else "-f"
        cmd = [
            f"{self.defects4j_path}/framework/bin/defects4j",
            "checkout",
            "-p", project,
            "-v", f"{bug_id}{version_flag}",
            "-w", str(target_dir)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _get_bug_metadata(self, project: str, bug_id: int) -> Dict:
        """Extract bug metadata from Defects4J"""
        cmd = [
            f"{self.defects4j_path}/framework/bin/defects4j",
            "info",
            "-p", project,
            "-b", str(bug_id)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output into dictionary
        metadata = {}
        for line in result.stdout.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip()] = value.strip()
        
        return metadata
    
    def _extract_test_suite(self, project_dir: Path) -> List[str]:
        """Extract test suite from project"""
        # Find all test files
        test_files = list(project_dir.rglob("*Test.java"))
        return [str(f.relative_to(project_dir)) for f in test_files]
    
    def get_all_bugs(self, project: str) -> List[int]:
        """Get list of all bug IDs for a project"""
        cmd = [
            f"{self.defects4j_path}/framework/bin/defects4j",
            "bids",
            "-p", project
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return [int(bid) for bid in result.stdout.strip().split('\n') if bid.strip()]

# Example usage function
def example_usage() -> BugInstance:
    """Example usage of the Defects4JLoader"""
    loader = Defects4JLoader()
    bug = loader.load_bug(project="Lang", bug_id=1)
    print(f"Loaded bug: {bug.project_id}-{bug.bug_id}")
    print(f"Buggy version: {bug.buggy_version_path}")
    print(f"Fixed version: {bug.fixed_version_path}")
    return bug

if __name__ == "__main__":
    example_usage()
