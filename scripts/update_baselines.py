#!/usr/bin/env python3
"""
Baseline Update Script

Semi-automated baseline updates with safeguards.
Only updates baselines on sustained improvements (5+ runs, >5% better).

Usage:
    # Interactive approval (recommended)
    python scripts/update_baselines.py

    # View proposed changes first
    python scripts/update_baselines.py --dry-run

    # Auto-approve improvements (CI)
    python scripts/update_baselines.py --auto-approve-improvements --min-runs 5

    # Specific platform
    python scripts/update_baselines.py --platform M1_Mac_16GB

Environment Variables:
    PERFORMANCE_DB_PATH: Path to performance database
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy not installed. Install with: pip install numpy")
    sys.exit(1)

from utils.performance_history import PerformanceHistory, TRACKED_METRICS
from utils.platform_detection import detect_platform, get_git_metadata, get_python_version

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Baselines file
BASELINES_FILE = Path(__file__).parent.parent / "tests" / "performance_baselines.json"


def load_baselines_file() -> Dict:
    """Load baselines from file."""
    if BASELINES_FILE.exists():
        with open(BASELINES_FILE) as f:
            return json.load(f)
    return {}


def save_baselines_file(baselines: Dict):
    """Save baselines to file."""
    with open(BASELINES_FILE, 'w') as f:
        json.dump(baselines, f, indent=2)
    log.info(f"Baselines saved to {BASELINES_FILE}")


def update_baselines(
    platform: Optional[str] = None,
    auto_approve: bool = False,
    min_runs: int = 5,
    dry_run: bool = False,
    improvement_threshold: float = 0.05
):
    """
    Update baselines based on recent performance improvements.

    Args:
        platform: Platform to update (None = auto-detect)
        auto_approve: Auto-approve improvements without prompts
        min_runs: Minimum consecutive runs showing improvement
        dry_run: Show changes without applying
        improvement_threshold: Minimum improvement to propose (default 5%)
    """
    log.info("=" * 70)
    log.info("Baseline Update Tool")
    log.info("=" * 70)

    # Initialize history
    history = PerformanceHistory()

    # Detect platform
    if platform is None:
        platform = detect_platform()
        log.info(f"Detected platform: {platform}")
    else:
        log.info(f"Using platform: {platform}")

    # Get recent runs
    recent_runs = history.get_recent_runs(platform, limit=min_runs * 2)

    if len(recent_runs) < min_runs:
        log.warning(f"Not enough runs ({len(recent_runs)} < {min_runs}). Need more data.")
        log.info("\nRun performance tests to collect more data:")
        log.info(f"  ENABLE_PERFORMANCE_RECORDING=1 pytest tests/test_performance_regression.py")
        return False

    log.info(f"Analyzing last {min_runs} runs...")

    # Load current baselines
    all_baselines = load_baselines_file()
    current_baselines = all_baselines.get(platform, {})

    if not current_baselines:
        log.warning(f"No current baselines for platform: {platform}")
        log.info("Creating initial baselines from recent runs...")
        current_baselines = {"metadata": {}}

    # Calculate proposed updates
    proposals = {}

    for metric in TRACKED_METRICS:
        # Get recent values for this metric
        values = [
            run.get(metric)
            for run in recent_runs[:min_runs]
            if run.get(metric) is not None
        ]

        if len(values) < min_runs:
            continue

        # Calculate median of recent runs
        proposed = float(np.median(values))
        current = current_baselines.get(metric)

        if current is None:
            # No baseline exists - propose this as initial baseline
            proposals[metric] = {
                "current": None,
                "proposed": proposed,
                "improvement": None,
                "type": "new"
            }
            continue

        # Calculate improvement
        # For latency metrics: lower is better
        # For throughput metrics: higher is better
        if metric in ["vector_search_latency", "query_latency_no_vllm",
                     "query_latency_vllm", "peak_memory_gb", "avg_memory_gb"]:
            # Lower is better
            improvement = (current - proposed) / current
        else:
            # Higher is better
            improvement = (proposed - current) / current

        # Only propose if improvement > threshold
        if improvement > improvement_threshold:
            proposals[metric] = {
                "current": current,
                "proposed": proposed,
                "improvement": improvement,
                "type": "improvement"
            }

    if not proposals:
        log.info("\n✅ No improvements detected. Baselines unchanged.")
        log.info("\nCurrent baselines are already optimal based on recent runs.")
        return True

    # Display proposals
    print(f"\n{'='*70}")
    print(f"Baseline Update Proposals for {platform}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Current':<12} {'Proposed':<12} {'Change':<10}")
    print(f"{'-'*70}")

    for metric, data in proposals.items():
        if data["type"] == "new":
            print(f"{metric:<30} {'(none)':<12} {data['proposed']:<12.3f} {'NEW':>10}")
        else:
            change_pct = data['improvement'] * 100
            print(f"{metric:<30} {data['current']:<12.3f} {data['proposed']:<12.3f} {change_pct:>8.1f}%")

    if dry_run:
        print("\n[DRY RUN] No changes applied.")
        return True

    # Approval
    if not auto_approve:
        print("\nApply these updates?")
        print("  yes - Apply all updates")
        print("  no  - Cancel")
        response = input("\nYour choice: ").strip().lower()

        if response not in ["yes", "y"]:
            log.info("Updates cancelled.")
            return False

    # Apply updates
    git_metadata = get_git_metadata()

    for metric, data in proposals.items():
        current_baselines[metric] = data["proposed"]

    # Update metadata
    current_baselines["metadata"] = {
        "platform": platform,
        "last_updated": datetime.now().isoformat(),
        "git_commit": git_metadata["commit"],
        "git_branch": git_metadata["branch"],
        "python_version": get_python_version(),
        "notes": f"Updated {len(proposals)} baseline(s) based on {min_runs} recent runs"
    }

    # Write back
    all_baselines[platform] = current_baselines
    save_baselines_file(all_baselines)

    print(f"\n✅ Updated {len(proposals)} baseline(s) for {platform}")
    print(f"\n📝 Next steps:")
    print(f"   1. Review the changes: git diff {BASELINES_FILE}")
    print(f"   2. Commit the changes:")
    print(f"      git add {BASELINES_FILE}")
    print(f"      git commit -m 'perf: update baselines for {platform}'")
    print(f"      git push")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update performance baselines"
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Platform identifier (default: auto-detect)"
    )
    parser.add_argument(
        "--auto-approve-improvements",
        action="store_true",
        help="Auto-approve improvements without prompts (for CI)"
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=5,
        help="Minimum consecutive runs showing improvement (default: 5)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed changes without applying"
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.05,
        help="Minimum improvement percentage to propose (default: 0.05 = 5%%)"
    )

    args = parser.parse_args()

    try:
        success = update_baselines(
            platform=args.platform,
            auto_approve=args.auto_approve_improvements,
            min_runs=args.min_runs,
            dry_run=args.dry_run,
            improvement_threshold=args.improvement_threshold
        )

        sys.exit(0 if success else 1)

    except Exception as e:
        log.error(f"Failed to update baselines: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
