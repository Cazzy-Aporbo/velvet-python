#!/usr/bin/env python3
"""
Check Python File Headers - Ensure proper attribution and documentation

Author: Cazzy Aporbo, MS
Created: January 2025

This script checks that all Python files have proper headers with:
- Module docstring
- Author attribution
- Consistent formatting

I created this after realizing half my files were missing proper attribution.
Now it runs automatically before every commit to ensure consistency.

Usage:
    python scripts/check_headers.py [files...]
    
If no files specified, checks all Python files in the project.
"""

import sys
import re
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
AUTHOR = "Cazzy Aporbo, MS"
REQUIRED_ELEMENTS = {
    'docstring': r'"""[\s\S]+?"""',
    'author': r'Author:\s*Cazzy Aporbo,?\s*MS',
    'created_or_updated': r'(Created|Updated|Last Updated):\s*\w+\s+\d{4}',
}

# Paths to exclude from checking
EXCLUDE_PATTERNS = [
    '**/migrations/**',
    '**/__pycache__/**',
    '**/build/**',
    '**/dist/**',
    '**/.venv/**',
    '**/venv/**',
    '**/env/**',
    '**/.git/**',
    '**/site/**',
    '**/htmlcov/**',
]

# Files that don't need full headers
MINIMAL_HEADER_FILES = [
    '__init__.py',
    'setup.py',
    'conftest.py',
]


@dataclass
class FileCheckResult:
    """
    Result of checking a single file.
    
    Author: Cazzy Aporbo, MS
    """
    path: Path
    has_docstring: bool
    has_author: bool
    has_date: bool
    issues: List[str]
    
    @property
    def is_valid(self) -> bool:
        """Check if file passes all requirements"""
        return self.has_docstring and self.has_author
    
    def __str__(self) -> str:
        """String representation of result"""
        status = "✓" if self.is_valid else "✗"
        return f"{status} {self.path}"


class HeaderChecker:
    """
    Checks Python files for proper headers.
    
    I use this to ensure every file is properly documented and attributed.
    It's part of my commitment to clear, maintainable code.
    
    Author: Cazzy Aporbo, MS
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the header checker.
        
        Args:
            verbose: Show detailed output
        """
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent
        
    def should_check_file(self, file_path: Path) -> bool:
        """
        Determine if a file should be checked.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file should be checked
        """
        # Skip excluded patterns
        for pattern in EXCLUDE_PATTERNS:
            if file_path.match(pattern):
                return False
        
        # Only check .py files
        if file_path.suffix != '.py':
            return False
        
        # Skip test files (they have different requirements)
        if 'test_' in file_path.name or file_path.name.endswith('_test.py'):
            return False
        
        return True
    
    def check_file(self, file_path: Path) -> FileCheckResult:
        """
        Check a single Python file for proper headers.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            FileCheckResult with validation details
        """
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            issues.append(f"Could not read file: {e}")
            return FileCheckResult(
                path=file_path,
                has_docstring=False,
                has_author=False,
                has_date=False,
                issues=issues
            )
        
        # For minimal header files, only check for author
        is_minimal = file_path.name in MINIMAL_HEADER_FILES
        
        # Check for module docstring (first thing in file)
        has_docstring = bool(re.match(r'^\s*"""', content))
        if not has_docstring and not is_minimal:
            issues.append("Missing module docstring")
        
        # Check for author attribution
        has_author = bool(re.search(REQUIRED_ELEMENTS['author'], content, re.IGNORECASE))
        if not has_author:
            if is_minimal:
                # For __init__.py, check if it's empty or minimal
                if len(content.strip()) > 10:  # Not empty
                    issues.append(f"Missing author attribution: {AUTHOR}")
            else:
                issues.append(f"Missing author attribution: {AUTHOR}")
        
        # Check for date (Created/Updated)
        has_date = bool(re.search(REQUIRED_ELEMENTS['created_or_updated'], content))
        if not has_date and not is_minimal and len(content.strip()) > 50:
            issues.append("Missing creation/update date")
        
        # Additional checks for main module files
        if str(file_path).startswith(str(self.project_root / 'velvet_python')):
            if not is_minimal:
                # Check docstring quality
                docstring_match = re.search(r'"""([\s\S]+?)"""', content)
                if docstring_match:
                    docstring = docstring_match.group(1)
                    if len(docstring.strip()) < 20:
                        issues.append("Docstring too short (add meaningful description)")
                    
                    # Should explain what the module does
                    if not any(word in docstring.lower() for word in ['this', 'module', 'class', 'function']):
                        issues.append("Docstring should explain what this module does")
        
        return FileCheckResult(
            path=file_path,
            has_docstring=has_docstring or is_minimal,
            has_author=has_author,
            has_date=has_date or is_minimal,
            issues=issues
        )
    
    def check_files(self, file_paths: Optional[List[Path]] = None) -> List[FileCheckResult]:
        """
        Check multiple files for proper headers.
        
        Args:
            file_paths: List of files to check (None = all project files)
            
        Returns:
            List of check results
        """
        if file_paths is None:
            # Find all Python files in project
            file_paths = []
            for pattern in ['**/*.py']:
                file_paths.extend(self.project_root.glob(pattern))
        
        results = []
        for file_path in file_paths:
            if self.should_check_file(file_path):
                result = self.check_file(file_path)
                results.append(result)
                
                if self.verbose:
                    print(f"Checking {file_path}... {'✓' if result.is_valid else '✗'}")
        
        return results
    
    def generate_header_template(self, file_path: Path) -> str:
        """
        Generate a proper header template for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Header template string
        """
        file_name = file_path.stem
        module_type = "Module"
        
        if "test_" in file_name:
            module_type = "Tests"
        elif file_name == "cli":
            module_type = "CLI"
        elif file_name == "__init__":
            return f'"""\n{file_path.parent.name.title()} Package\n\nAuthor: {AUTHOR}\n"""\n'
        
        template = f'''"""
{file_name.replace('_', ' ').title()} - Brief description here

Author: {AUTHOR}
Created: January 2025

Longer description of what this module does and why it exists.
Add any important notes about usage or implementation.
"""

'''
        return template
    
    def fix_file(self, file_path: Path, result: FileCheckResult) -> bool:
        """
        Attempt to fix header issues in a file.
        
        Args:
            file_path: Path to file
            result: Check result for the file
            
        Returns:
            True if file was fixed
        """
        if result.is_valid:
            return True
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # If missing docstring entirely, add one
            if not result.has_docstring and file_path.name not in MINIMAL_HEADER_FILES:
                header = self.generate_header_template(file_path)
                content = header + content
            
            # If missing author, try to add it to existing docstring
            elif not result.has_author:
                # Find the docstring
                docstring_match = re.search(r'("""[\s\S]+?""")', content)
                if docstring_match:
                    docstring = docstring_match.group(1)
                    # Add author before closing """
                    new_docstring = docstring[:-3] + f"\nAuthor: {AUTHOR}\n" + '"""'
                    content = content.replace(docstring, new_docstring)
                else:
                    # Add minimal docstring for special files
                    header = self.generate_header_template(file_path)
                    content = header + content
            
            # Write back
            file_path.write_text(content, encoding='utf-8')
            
            if self.verbose:
                print(f"Fixed: {file_path}")
            
            return True
            
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False
    
    def print_report(self, results: List[FileCheckResult]) -> None:
        """
        Print a summary report of check results.
        
        Args:
            results: List of check results
        """
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid
        
        print("\n" + "="*60)
        print(f"Python File Header Check Report")
        print(f"Author: {AUTHOR}")
        print("="*60)
        
        if invalid > 0:
            print(f"\n❌ Found {invalid} file(s) with issues:\n")
            
            for result in results:
                if not result.is_valid:
                    relative_path = result.path.relative_to(self.project_root)
                    print(f"  ✗ {relative_path}")
                    for issue in result.issues:
                        print(f"    - {issue}")
        
        print(f"\n📊 Summary: {valid}/{total} files have proper headers")
        
        if invalid == 0:
            print("✅ All files have proper headers! Great job!")
        else:
            print(f"❌ {invalid} file(s) need attention")
            print("\nTo fix: Add proper docstrings and author attribution")
            print(f"Required: Author: {AUTHOR}")


def main():
    """
    Main entry point for header checking.
    
    Author: Cazzy Aporbo, MS
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"Check Python file headers - by {AUTHOR}"
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Files to check (default: all Python files)'
    )
    parser.add_argument(
        '--fix',
        action='store_true',
        help='Attempt to fix header issues automatically'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output'
    )
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = HeaderChecker(verbose=args.verbose)
    
    # Get files to check
    if args.files:
        file_paths = [Path(f) for f in args.files]
    else:
        file_paths = None
    
    # Run checks
    results = checker.check_files(file_paths)
    
    # Fix if requested
    if args.fix:
        print("Attempting to fix header issues...")
        fixed = 0
        for result in results:
            if not result.is_valid:
                if checker.fix_file(result.path, result):
                    fixed += 1
        
        if fixed > 0:
            print(f"✅ Fixed {fixed} file(s)")
            # Re-check to verify
            results = checker.check_files(file_paths)
    
    # Print report
    checker.print_report(results)
    
    # Exit with error if any files are invalid
    invalid_count = sum(1 for r in results if not r.is_valid)
    if invalid_count > 0:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
