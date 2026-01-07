#!/usr/bin/env python3
"""
Repository Organization Validator

Audits repository structure against industry best practices and provides
actionable recommendations for improvement.

Usage:
    python scripts/validate_repo_organization.py [--format=text|json|markdown]

Exit codes:
    0: Repository well-organized (score >= 8.0)
    1: Repository needs improvement (score < 8.0)
    2: Critical issues found (missing LICENSE, secrets in repo)
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class Priority(Enum):
    """Issue priority levels."""
    CRITICAL = "P0"
    HIGH = "P1"
    MEDIUM = "P2"
    LOW = "P3"


@dataclass
class Issue:
    """Represents an organization issue."""
    priority: Priority
    category: str
    title: str
    description: str
    fix: str
    effort: str  # Low, Medium, High

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['priority'] = self.priority.value
        return d


class RepoValidator:
    """Validates repository organization."""

    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path.resolve()
        self.issues: List[Issue] = []
        self.strengths: List[str] = []
        self.score = 10.0

    def run_audit(self) -> Dict:
        """Run complete audit and return results."""
        print("🔍 Analyzing repository structure...")

        # Detect language/framework
        self.language = self._detect_language()
        print(f"   Language detected: {self.language}")

        # Run all checks
        self._check_essential_files()
        self._check_root_directory()
        self._check_directory_structure()
        self._check_documentation()
        self._check_gitignore()
        self._check_dependency_management()
        self._check_build_automation()
        self._check_git_hygiene()

        # Calculate final score
        self._calculate_score()

        return {
            'score': round(self.score, 1),
            'language': self.language,
            'issues': [issue.to_dict() for issue in self.issues],
            'strengths': self.strengths,
            'summary': self._generate_summary()
        }

    def _detect_language(self) -> str:
        """Detect primary language/framework."""
        if (self.repo_path / "pyproject.toml").exists() or \
           (self.repo_path / "requirements.txt").exists() or \
           (self.repo_path / "setup.py").exists():
            return "Python"
        elif (self.repo_path / "package.json").exists():
            return "JavaScript/TypeScript"
        elif (self.repo_path / "go.mod").exists():
            return "Go"
        elif (self.repo_path / "Cargo.toml").exists():
            return "Rust"
        elif (self.repo_path / "pom.xml").exists():
            return "Java"
        else:
            return "Unknown"

    def _check_essential_files(self):
        """Check for essential files."""
        essential = {
            'README.md': (Priority.CRITICAL, "Project overview required"),
            'LICENSE': (Priority.CRITICAL, "Legal requirement for OSS"),
            '.gitignore': (Priority.CRITICAL, "Prevent accidental commits"),
            'CONTRIBUTING.md': (Priority.HIGH, "Contribution guidelines"),
        }

        for file, (priority, desc) in essential.items():
            path = self.repo_path / file
            if not path.exists():
                self.issues.append(Issue(
                    priority=priority,
                    category="Essential Files",
                    title=f"Missing {file}",
                    description=desc,
                    fix=f"Create {file} with proper template",
                    effort="Low"
                ))
            else:
                self.strengths.append(f"{file} present")

    def _check_root_directory(self):
        """Check root directory clutter."""
        root_files = [
            f for f in self.repo_path.iterdir()
            if f.is_file() and not f.name.startswith('.')
        ]

        if len(root_files) > 15:
            self.issues.append(Issue(
                priority=Priority.HIGH,
                category="Organization",
                title=f"Root directory clutter ({len(root_files)} files)",
                description="Root should have <15 files for clarity",
                fix="Move docs to docs/, configs to config/, scripts to scripts/",
                effort="Medium"
            ))
        elif len(root_files) <= 12:
            self.strengths.append(f"Clean root directory ({len(root_files)} files)")

    def _check_directory_structure(self):
        """Check standard directory presence."""
        standard_dirs = {
            'tests': (Priority.HIGH, "Test suite directory"),
            'docs': (Priority.MEDIUM, "Documentation directory"),
            'config': (Priority.MEDIUM, "Configuration directory"),
            'scripts': (Priority.LOW, "Automation scripts"),
        }

        # Language-specific source directory
        if self.language == "Python":
            src_patterns = ['src/', 'lib/', self.repo_path.name]
        elif self.language in ["JavaScript/TypeScript", "Go", "Rust"]:
            src_patterns = ['src/', 'lib/']
        else:
            src_patterns = ['src/']

        has_src = any((self.repo_path / p).exists() for p in src_patterns)
        if not has_src:
            self.issues.append(Issue(
                priority=Priority.HIGH,
                category="Structure",
                title="No source directory (src/ or lib/)",
                description="Source code should be in dedicated directory",
                fix="Create src/ or lib/ and organize code",
                effort="Medium"
            ))
        else:
            self.strengths.append("Source code organized in dedicated directory")

        for dir_name, (priority, desc) in standard_dirs.items():
            path = self.repo_path / dir_name
            if not path.exists():
                self.issues.append(Issue(
                    priority=priority,
                    category="Structure",
                    title=f"Missing {dir_name}/ directory",
                    description=desc,
                    fix=f"Create {dir_name}/ directory",
                    effort="Low"
                ))
            else:
                self.strengths.append(f"{dir_name}/ directory present")

    def _check_documentation(self):
        """Check documentation quality."""
        readme_path = self.repo_path / "README.md"

        if readme_path.exists():
            content = readme_path.read_text()

            # Check README sections
            required_sections = ['Quick Start', 'Install', 'Usage', 'Features']
            missing = [s for s in required_sections if s.lower() not in content.lower()]

            if missing:
                self.issues.append(Issue(
                    priority=Priority.MEDIUM,
                    category="Documentation",
                    title=f"README missing sections: {', '.join(missing)}",
                    description="README should include quick start, features, usage",
                    fix="Add missing sections to README",
                    effort="Low"
                ))
            else:
                self.strengths.append("README has comprehensive sections")

        # Check for docs/ organization
        docs_dir = self.repo_path / "docs"
        if docs_dir.exists():
            doc_files = list(docs_dir.glob("*.md"))
            if len(doc_files) >= 3:
                self.strengths.append(f"Documentation organized ({len(doc_files)} guides)")

    def _check_gitignore(self):
        """Check .gitignore completeness."""
        gitignore_path = self.repo_path / ".gitignore"

        if not gitignore_path.exists():
            return  # Already flagged in essential files

        content = gitignore_path.read_text()

        # Common patterns by language
        python_patterns = ['__pycache__', '.venv', '*.pyc', '.pytest_cache']
        js_patterns = ['node_modules', 'npm-debug.log', 'dist/']
        common_patterns = ['.env', '*.log', '.DS_Store']

        missing_patterns = []

        if self.language == "Python":
            missing_patterns = [p for p in python_patterns if p not in content]
        elif self.language == "JavaScript/TypeScript":
            missing_patterns = [p for p in js_patterns if p not in content]

        missing_common = [p for p in common_patterns if p not in content]
        missing_patterns.extend(missing_common)

        if missing_patterns:
            self.issues.append(Issue(
                priority=Priority.MEDIUM,
                category="Git Hygiene",
                title=f"Incomplete .gitignore (missing {len(missing_patterns)} patterns)",
                description=f"Missing patterns: {', '.join(missing_patterns[:3])}...",
                fix="Enhance .gitignore with common patterns",
                effort="Low"
            ))
        else:
            self.strengths.append(".gitignore comprehensive")

    def _check_dependency_management(self):
        """Check dependency management practices."""
        if self.language == "Python":
            has_pyproject = (self.repo_path / "pyproject.toml").exists()
            has_requirements = (self.repo_path / "requirements.txt").exists()

            if not has_pyproject and not has_requirements:
                self.issues.append(Issue(
                    priority=Priority.CRITICAL,
                    category="Dependencies",
                    title="No dependency management",
                    description="Python project needs pyproject.toml or requirements.txt",
                    fix="Create pyproject.toml with dependencies",
                    effort="Low"
                ))
            elif not has_pyproject:
                self.issues.append(Issue(
                    priority=Priority.HIGH,
                    category="Dependencies",
                    title="Missing pyproject.toml (modern standard)",
                    description="PEP 518 recommends pyproject.toml over setup.py",
                    fix="Create pyproject.toml with project metadata",
                    effort="Low"
                ))
            else:
                self.strengths.append("Modern dependency management (pyproject.toml)")

            # Check for split requirements
            has_dev = (self.repo_path / "requirements-dev.txt").exists()
            if has_requirements and not has_dev and not has_pyproject:
                self.issues.append(Issue(
                    priority=Priority.MEDIUM,
                    category="Dependencies",
                    title="Dependencies not split (core vs dev)",
                    description="Separate dev dependencies for cleaner production installs",
                    fix="Create requirements-dev.txt for dev tools",
                    effort="Low"
                ))

    def _check_build_automation(self):
        """Check for build automation."""
        automation_files = ['Makefile', 'justfile', 'package.json']

        has_automation = any((self.repo_path / f).exists() for f in automation_files)

        if not has_automation:
            self.issues.append(Issue(
                priority=Priority.MEDIUM,
                category="Automation",
                title="No build automation (Makefile)",
                description="Common tasks should be automated",
                fix="Create Makefile with install, test, lint targets",
                effort="Medium"
            ))
        else:
            self.strengths.append("Build automation present")

    def _check_git_hygiene(self):
        """Check git repository health."""
        try:
            # Check for .env files tracked
            result = subprocess.run(
                ['git', 'ls-files', '*.env'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.stdout.strip():
                self.issues.append(Issue(
                    priority=Priority.CRITICAL,
                    category="Security",
                    title="Secrets potentially committed (.env files tracked)",
                    description="Environment files should not be in git",
                    fix="Remove .env from git, add to .gitignore",
                    effort="Low"
                ))

            # Check for large files
            result = subprocess.run(
                ['git', 'ls-files', '--', '*.bin', '*.model', '*.ckpt'],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.stdout.strip():
                self.issues.append(Issue(
                    priority=Priority.HIGH,
                    category="Git Hygiene",
                    title="Large binary files in git",
                    description="Large files should use Git LFS",
                    fix="Use Git LFS for binary files or move to external storage",
                    effort="Medium"
                ))

        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass  # Git not available or command failed

    def _calculate_score(self):
        """Calculate overall organization score (0-10)."""
        # Start at 10, deduct points for issues
        deductions = {
            Priority.CRITICAL: 2.0,
            Priority.HIGH: 1.0,
            Priority.MEDIUM: 0.3,
            Priority.LOW: 0.1
        }

        for issue in self.issues:
            self.score -= deductions.get(issue.priority, 0)

        self.score = max(0.0, self.score)

    def _generate_summary(self) -> Dict:
        """Generate audit summary."""
        priority_counts = {p: 0 for p in Priority}
        for issue in self.issues:
            priority_counts[issue.priority] += 1

        return {
            'total_issues': len(self.issues),
            'critical': priority_counts[Priority.CRITICAL],
            'high': priority_counts[Priority.HIGH],
            'medium': priority_counts[Priority.MEDIUM],
            'low': priority_counts[Priority.LOW],
            'strengths': len(self.strengths)
        }

    def format_text(self, results: Dict) -> str:
        """Format results as text."""
        lines = []
        lines.append("=" * 70)
        lines.append("REPOSITORY ORGANIZATION AUDIT")
        lines.append("=" * 70)
        lines.append(f"\n📊 Overall Score: {results['score']}/10.0")
        lines.append(f"🔧 Language: {results['language']}")

        summary = results['summary']
        lines.append(f"\n📈 Issues Found: {summary['total_issues']}")
        lines.append(f"   Critical (P0): {summary['critical']}")
        lines.append(f"   High (P1): {summary['high']}")
        lines.append(f"   Medium (P2): {summary['medium']}")
        lines.append(f"   Low (P3): {summary['low']}")

        lines.append(f"\n✅ Strengths: {summary['strengths']}")

        if results['strengths']:
            lines.append("\nWhat's Working Well:")
            for strength in results['strengths']:
                lines.append(f"  ✓ {strength}")

        if results['issues']:
            lines.append("\n" + "=" * 70)
            lines.append("ISSUES & RECOMMENDATIONS")
            lines.append("=" * 70)

            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
                priority_issues = [i for i in results['issues'] if i['priority'] == priority.value]

                if priority_issues:
                    lines.append(f"\n### {priority.value} - {priority.name} PRIORITY ###")
                    for idx, issue in enumerate(priority_issues, 1):
                        lines.append(f"\n{idx}. {issue['title']}")
                        lines.append(f"   Category: {issue['category']}")
                        lines.append(f"   Issue: {issue['description']}")
                        lines.append(f"   Fix: {issue['fix']}")
                        lines.append(f"   Effort: {issue['effort']}")

        lines.append("\n" + "=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)

        critical_count = summary['critical']
        if critical_count > 0:
            lines.append(f"\n⚠️  CRITICAL: Fix {critical_count} critical issue(s) immediately")
            lines.append("   Run: /organize-repo --mode=quick-fix")

        if results['score'] < 5.0:
            lines.append("\n⚠️  Score below 5.0: Full reorganization recommended")
            lines.append("   Run: /organize-repo --mode=full")
        elif results['score'] < 8.0:
            lines.append("\n💡 Score below 8.0: Apply recommended improvements")
            lines.append("   Review high-priority issues above")
        else:
            lines.append("\n✅ Repository well-organized! Minor improvements suggested.")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def format_markdown(self, results: Dict) -> str:
        """Format results as markdown."""
        lines = []
        lines.append("# Repository Organization Audit Report")
        lines.append(f"\n**Generated**: {self._get_timestamp()}")
        lines.append(f"**Score**: {results['score']}/10.0")
        lines.append(f"**Language**: {results['language']}")

        summary = results['summary']
        lines.append("\n## Summary")
        lines.append(f"- Total Issues: {summary['total_issues']}")
        lines.append(f"- Critical (P0): {summary['critical']}")
        lines.append(f"- High (P1): {summary['high']}")
        lines.append(f"- Medium (P2): {summary['medium']}")
        lines.append(f"- Low (P3): {summary['low']}")
        lines.append(f"- Strengths: {summary['strengths']}")

        if results['strengths']:
            lines.append("\n## Strengths ✅")
            for strength in results['strengths']:
                lines.append(f"- {strength}")

        if results['issues']:
            lines.append("\n## Issues Found")

            for priority in [Priority.CRITICAL, Priority.HIGH, Priority.MEDIUM, Priority.LOW]:
                priority_issues = [i for i in results['issues'] if i['priority'] == priority.value]

                if priority_issues:
                    emoji = {"P0": "🔴", "P1": "🟠", "P2": "🟡", "P3": "🔵"}
                    lines.append(f"\n### {emoji.get(priority.value, '')} {priority.value} - {priority.name} Priority")

                    for idx, issue in enumerate(priority_issues, 1):
                        lines.append(f"\n**{idx}. {issue['title']}**")
                        lines.append(f"- **Category**: {issue['category']}")
                        lines.append(f"- **Issue**: {issue['description']}")
                        lines.append(f"- **Fix**: {issue['fix']}")
                        lines.append(f"- **Effort**: {issue['effort']}")

        lines.append("\n## Recommendations")

        if summary['critical'] > 0:
            lines.append(f"\n### Immediate Actions Required")
            lines.append(f"Fix {summary['critical']} critical issue(s) before proceeding.")

        if results['score'] < 8.0:
            lines.append("\n### Suggested Next Steps")
            lines.append("1. Review high-priority issues")
            lines.append("2. Run `/organize-repo --mode=quick-fix` for quick wins")
            lines.append("3. Plan full reorganization with `/organize-repo --mode=full`")

        return "\n".join(lines)

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Main entry point."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Validate repository organization")
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'markdown'],
        default='text',
        help='Output format'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file (default: stdout)'
    )

    args = parser.parse_args()

    # Run validation
    validator = RepoValidator()
    results = validator.run_audit()

    # Format output
    if args.format == 'json':
        output = json.dumps(results, indent=2)
    elif args.format == 'markdown':
        output = validator.format_markdown(results)
    else:
        output = validator.format_text(results)

    # Write output
    if args.output:
        args.output.write_text(output)
        print(f"✅ Report written to {args.output}")
    else:
        print(output)

    # Exit code based on score
    if results['summary']['critical'] > 0:
        sys.exit(2)  # Critical issues
    elif results['score'] < 8.0:
        sys.exit(1)  # Needs improvement
    else:
        sys.exit(0)  # Good organization


if __name__ == '__main__':
    main()
