#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner

Run full benchmark suite for nightly CI.
This script orchestrates all performance tests and generates reports.

Usage:
    python scripts/run_comprehensive_benchmark.py --output benchmarks/nightly/20260107

Environment Variables:
    PERFORMANCE_RUN_TYPE: Run type (nightly, manual)
    BENCHMARK_MODE: Benchmark mode (quick, standard, comprehensive)
    ENABLE_PERFORMANCE_RECORDING: Enable recording to history database
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


def run_command(cmd: list, description: str, env: dict = None) -> bool:
    """
    Run a command and return success status.

    Args:
        cmd: Command to run (list of arguments)
        description: Description for logging
        env: Optional environment variables

    Returns:
        True if command succeeded, False otherwise
    """
    log.info(f"Running: {description}")

    # Merge environment
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=cmd_env,
            timeout=600  # 10 minute timeout
        )

        log.info(f"✅ {description} completed successfully")
        if result.stdout:
            log.debug(f"Output: {result.stdout[:500]}")  # First 500 chars

        return True

    except subprocess.CalledProcessError as e:
        log.error(f"❌ {description} failed with exit code {e.returncode}")
        if e.stdout:
            log.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            log.error(f"STDERR: {e.stderr}")
        return False

    except subprocess.TimeoutExpired:
        log.error(f"❌ {description} timed out")
        return False

    except Exception as e:
        log.error(f"❌ {description} failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive performance benchmarks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"benchmarks/nightly/{datetime.now().strftime('%Y%m%d')}",
        help="Output directory for reports"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "comprehensive"],
        default=None,
        help="Benchmark mode (default: from env or standard)"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # Determine mode
    mode = args.mode or os.getenv("BENCHMARK_MODE", "standard")
    log.info(f"Benchmark mode: {mode}")

    # Environment for commands
    env = {
        "ENABLE_PERFORMANCE_RECORDING": "1",
        "PERFORMANCE_RUN_TYPE": os.getenv("PERFORMANCE_RUN_TYPE", "nightly"),
    }

    results = {}

    # ========================================================================
    # 1. Run Performance Regression Tests
    # ========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 1: Performance Regression Tests")
    log.info("=" * 70)

    pytest_args = [
        "pytest",
        "tests/test_performance_regression.py",
        "-v",
        "--tb=short",
        "--no-cov",  # Disable coverage for benchmarks
        f"--junit-xml={output_dir}/regression-results.xml"
    ]

    # Add markers based on mode
    if mode == "quick":
        pytest_args.extend(["-m", "not slow"])
    elif mode == "comprehensive":
        # Run all tests including slow ones
        pass
    else:  # standard
        pytest_args.extend(["-m", "not slow"])

    results["regression_tests"] = run_command(
        pytest_args,
        "Performance regression tests",
        env=env
    )

    # ========================================================================
    # 2. Run RAG Benchmark (if available)
    # ========================================================================
    if mode in ["standard", "comprehensive"]:
        log.info("\n" + "=" * 70)
        log.info("STEP 2: RAG Quality Benchmarks")
        log.info("=" * 70)

        # Check if RAG benchmark script exists
        rag_benchmark_script = Path("scripts/benchmark_rag_quality.py")

        if rag_benchmark_script.exists():
            results["rag_benchmark"] = run_command(
                [
                    "python",
                    str(rag_benchmark_script),
                    "--output",
                    str(output_dir / "rag_benchmark.json")
                ],
                "RAG quality benchmark",
                env=env
            )
        else:
            log.warning("RAG benchmark script not found, skipping")
            results["rag_benchmark"] = None

    # ========================================================================
    # 3. Generate Reports
    # ========================================================================
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Generate Reports")
    log.info("=" * 70)

    # HTML report
    results["html_report"] = run_command(
        [
            "python",
            "scripts/generate_performance_report.py",
            "--format",
            "html",
            "--output",
            str(output_dir / "report.html")
        ],
        "HTML performance report"
    )

    # Markdown report
    results["markdown_report"] = run_command(
        [
            "python",
            "scripts/generate_performance_report.py",
            "--format",
            "markdown",
            "--output",
            str(output_dir / "summary.md")
        ],
        "Markdown performance report"
    )

    # JSON report
    results["json_report"] = run_command(
        [
            "python",
            "scripts/generate_performance_report.py",
            "--format",
            "json",
            "--output",
            str(output_dir / "metrics.json")
        ],
        "JSON performance report"
    )

    # ========================================================================
    # 4. Summary
    # ========================================================================
    log.info("\n" + "=" * 70)
    log.info("BENCHMARK SUMMARY")
    log.info("=" * 70)

    for step, success in results.items():
        if success is None:
            status = "⏭️  SKIPPED"
        elif success:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"

        log.info(f"  {step:30s} {status}")

    # Overall result
    failed_steps = [k for k, v in results.items() if v is False]

    if failed_steps:
        log.error(f"\n❌ Benchmark failed: {len(failed_steps)} step(s) failed")
        log.error(f"Failed steps: {', '.join(failed_steps)}")
        sys.exit(1)
    else:
        log.info("\n✅ All benchmarks completed successfully")
        log.info(f"📁 Results saved to: {output_dir}")
        sys.exit(0)


if __name__ == "__main__":
    main()
