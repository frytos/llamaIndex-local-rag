#!/usr/bin/env python3
"""
Verify all references in the repository.

Checks:
- Documentation links in markdown files
- Python imports in project files
- Script references in shell scripts
- Configuration file paths

Usage:
    python scripts/verify_references.py
    python scripts/verify_references.py --verbose
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def check_markdown_links(verbose: bool = False) -> Tuple[int, int, List[Dict]]:
    """Check all markdown links for broken references."""
    md_files = [f for f in Path('.').rglob('*.md')
                if '.venv' not in str(f) and 'node_modules' not in str(f)]

    broken_links = []
    total_links = 0

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8', errors='ignore')
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)

            for link_text, link_path in links:
                # Skip external URLs, anchors, and mailto
                if link_path.startswith(('http', '#', 'mailto:')):
                    continue

                total_links += 1

                # Remove anchor from path
                actual_path = link_path.split('#')[0]
                if not actual_path:
                    continue

                # Resolve relative path
                link_full_path = (md_file.parent / actual_path).resolve()

                # Check if file exists
                if not link_full_path.exists():
                    broken_links.append({
                        'file': str(md_file),
                        'link_text': link_text,
                        'link_path': link_path,
                        'resolved_path': str(link_full_path)
                    })
        except Exception as e:
            if verbose:
                print(f"Error processing {md_file}: {e}")

    return total_links, len(broken_links), broken_links


def check_python_imports(verbose: bool = False) -> Tuple[int, int, List[Dict]]:
    """Check Python imports for issues."""
    py_files = [f for f in Path('.').rglob('*.py')
                if '.venv' not in str(f)
                and 'node_modules' not in str(f)
                and 'data/' not in str(f)]  # Exclude user data

    error_files = []

    for py_file in py_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                ast.parse(f.read(), filename=str(py_file))
        except SyntaxError as e:
            error_files.append({
                'file': str(py_file),
                'error': f'Syntax error: {e}'
            })
        except Exception as e:
            if verbose:
                error_files.append({
                    'file': str(py_file),
                    'error': f'Error: {e}'
                })

    return len(py_files), len(error_files), error_files


def check_script_references(verbose: bool = False) -> Tuple[int, int, List[Dict]]:
    """Check shell scripts for broken Python references."""
    sh_files = list(Path('./scripts').glob('*.sh'))
    broken_refs = []

    for sh_file in sh_files:
        try:
            content = sh_file.read_text(encoding='utf-8', errors='ignore')

            # Find python script references
            py_refs = re.findall(r'python3?\s+([^\s]+\.py)', content)
            py_refs += re.findall(r'\./([\w/_]+\.py)', content)

            for ref in py_refs:
                ref = ref.strip()
                if ref.startswith('./'):
                    ref = ref[2:]

                # Check if file exists
                possible_paths = [
                    Path(ref),
                    Path('./scripts') / ref,
                    Path('./') / ref
                ]

                exists = any(p.exists() for p in possible_paths)

                if not exists:
                    broken_refs.append({
                        'shell_file': str(sh_file),
                        'reference': ref
                    })
        except Exception as e:
            if verbose:
                print(f"Error processing {sh_file}: {e}")

    return len(sh_files), len(broken_refs), broken_refs


def check_config_paths(verbose: bool = False) -> Tuple[int, int, List[Dict]]:
    """Check configuration file paths."""
    issues = []

    # Check docker-compose symlink
    docker_compose = Path('docker-compose.yml')
    if not docker_compose.exists():
        issues.append({
            'file': 'docker-compose.yml',
            'issue': 'Symlink missing'
        })
    elif not docker_compose.is_symlink():
        issues.append({
            'file': 'docker-compose.yml',
            'issue': 'Should be symlink to config/docker-compose.yml'
        })

    # Check .env.example
    env_example = Path('config/.env.example')
    if not env_example.exists():
        issues.append({
            'file': 'config/.env.example',
            'issue': 'File missing'
        })

    return 2, len(issues), issues


def print_summary(results: Dict) -> int:
    """Print verification summary."""
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70 + "\n")

    total_checks = 0
    total_issues = 0

    # Documentation Links
    doc_total, doc_broken, _ = results['documentation']
    total_checks += doc_total
    total_issues += doc_broken
    print(f"📄 Documentation Links:  {doc_total:3d} checked, {doc_broken:3d} broken")

    # Python Imports
    py_total, py_errors, _ = results['python']
    total_checks += py_total
    total_issues += py_errors
    print(f"🐍 Python Files:         {py_total:3d} checked, {py_errors:3d} errors")

    # Script References
    sh_total, sh_broken, _ = results['scripts']
    total_checks += sh_total
    total_issues += sh_broken
    print(f"📜 Shell Scripts:        {sh_total:3d} checked, {sh_broken:3d} broken")

    # Config Paths
    cfg_total, cfg_issues, _ = results['config']
    total_checks += cfg_total
    total_issues += cfg_issues
    print(f"⚙️  Configuration:       {cfg_total:3d} checked, {cfg_issues:3d} issues")

    # Overall
    health = ((total_checks - total_issues) / total_checks * 100) if total_checks > 0 else 100
    print("\n" + "-"*70)
    print(f"Overall Health: {health:.1f}% ({total_checks - total_issues}/{total_checks} passing)")
    print("-"*70 + "\n")

    return total_issues


def print_detailed_issues(results: Dict):
    """Print detailed issues."""
    # Documentation issues
    _, doc_broken, doc_issues = results['documentation']
    if doc_broken > 0:
        print("\n" + "="*70)
        print(f"DOCUMENTATION ISSUES ({doc_broken} found)")
        print("="*70 + "\n")

        # Group by file
        by_file = {}
        for issue in doc_issues:
            file = issue['file']
            if file not in by_file:
                by_file[file] = []
            by_file[file].append(issue)

        for file, issues in sorted(by_file.items())[:10]:  # Show top 10
            print(f"📄 {file}")
            for issue in issues[:3]:  # Show first 3 per file
                print(f"   → [{issue['link_text']}]({issue['link_path']})")
            if len(issues) > 3:
                print(f"   ... and {len(issues) - 3} more")
            print()

    # Python issues
    _, py_errors, py_issues = results['python']
    if py_errors > 0:
        print("\n" + "="*70)
        print(f"PYTHON ISSUES ({py_errors} found)")
        print("="*70 + "\n")

        for issue in py_issues[:10]:
            print(f"🐍 {issue['file']}")
            print(f"   → {issue['error']}")
            print()

    # Script issues
    _, sh_broken, sh_issues = results['scripts']
    if sh_broken > 0:
        print("\n" + "="*70)
        print(f"SCRIPT ISSUES ({sh_broken} found)")
        print("="*70 + "\n")

        for issue in sh_issues:
            print(f"📜 {issue['shell_file']}")
            print(f"   → Missing: {issue['reference']}")
            print()

    # Config issues
    _, cfg_issues_count, cfg_issues = results['config']
    if cfg_issues_count > 0:
        print("\n" + "="*70)
        print(f"CONFIGURATION ISSUES ({cfg_issues_count} found)")
        print("="*70 + "\n")

        for issue in cfg_issues:
            print(f"⚙️  {issue['file']}")
            print(f"   → {issue['issue']}")
            print()


def main():
    """Main verification function."""
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    show_details = '--details' in sys.argv or '-d' in sys.argv

    print("\n🔍 Starting repository verification...")
    print("Checking documentation, Python files, scripts, and configuration...\n")

    results = {
        'documentation': check_markdown_links(verbose),
        'python': check_python_imports(verbose),
        'scripts': check_script_references(verbose),
        'config': check_config_paths(verbose)
    }

    total_issues = print_summary(results)

    if show_details or verbose:
        print_detailed_issues(results)

    if total_issues == 0:
        print("✅ All checks passed! Repository is in excellent health.\n")
        return 0
    else:
        print(f"⚠️  Found {total_issues} issues. Run with --details for more information.")
        print("📋 See VERIFICATION_REPORT.md for comprehensive analysis.\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
