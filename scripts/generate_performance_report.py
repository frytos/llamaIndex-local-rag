#!/usr/bin/env python3
"""
Performance Report Generator

Generate markdown/HTML/JSON reports comparing current performance to baselines.
Used in CI/CD pipelines to post PR comments and create artifacts.

Usage:
    python scripts/generate_performance_report.py --format markdown --output report.md
    python scripts/generate_performance_report.py --format html --output report.html
    python scripts/generate_performance_report.py --format json --output report.json

Environment Variables:
    PERFORMANCE_DB_PATH: Path to performance database
    REGRESSION_THRESHOLD: Regression detection threshold (default 0.20)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.performance_history import PerformanceHistory
from utils.platform_detection import detect_platform, get_platform_info

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def format_value(metric: str, value: float) -> str:
    """
    Format metric value for display.

    Args:
        metric: Metric name
        value: Metric value

    Returns:
        Formatted string (e.g., "8.00s", "67.0 c/s", "42%")
    """
    # Latency metrics (seconds)
    if "latency" in metric or "time" in metric:
        return f"{value:.3f}s"

    # Memory metrics (GB)
    if "memory" in metric:
        return f"{value:.2f} GB"

    # Rate metrics (percentage)
    if "rate" in metric:
        return f"{value*100:.1f}%"

    # Throughput metrics
    if "throughput" in metric:
        if "embedding" in metric:
            return f"{value:.1f} c/s"  # chunks/sec
        elif "db_insertion" in metric:
            return f"{value:.0f} n/s"  # nodes/sec
        else:
            return f"{value:.1f}/s"

    # Tokens per second
    if "tokens" in metric:
        return f"{value:.1f} t/s"

    # Quality metrics (0-1 scale)
    if "mrr" in metric or "ndcg" in metric:
        return f"{value:.3f}"

    # Default
    return f"{value:.3f}"


def generate_markdown_report(
    run: Dict, comparison: Dict, platform: str, threshold: float
) -> str:
    """
    Generate markdown report for PR comments.

    Args:
        run: Latest performance run
        comparison: Comparison results from compare_to_baseline()
        platform: Platform identifier
        threshold: Regression threshold

    Returns:
        Markdown-formatted report
    """
    lines = [
        "## 📊 Performance Report",
        "",
        f"**Platform**: `{platform}`",
        f"**Timestamp**: {run.get('timestamp', 'N/A')}",
    ]

    # Add git metadata if available
    if run.get("git_commit"):
        lines.append(f"**Commit**: `{run['git_commit']}`")
    if run.get("git_branch"):
        lines.append(f"**Branch**: `{run['git_branch']}`")

    lines.extend(["", "### 📈 Metrics Comparison", ""])

    # Check if we have any metrics to compare
    has_metrics = (
        comparison["regressions"] or comparison["improvements"] or comparison["stable"]
    )

    if not has_metrics:
        lines.append("⚠️ No baseline metrics available for comparison.")
        lines.append("")
        lines.append(
            "_This is likely the first run. Future runs will be compared against this baseline._"
        )
        return "\n".join(lines)

    # Build table
    lines.extend(
        [
            "| Metric | Current | Baseline | Change | Status |",
            "|--------|---------|----------|--------|--------|",
        ]
    )

    # Stable metrics (within threshold)
    for metric, current, baseline, pct in comparison["stable"]:
        status = "✅"
        change = f"{pct:+.1%}"
        current_str = format_value(metric, current)
        baseline_str = format_value(metric, baseline)
        lines.append(
            f"| {metric} | {current_str} | {baseline_str} | {change} | {status} |"
        )

    # Improvements (>5% better)
    for metric, current, baseline, pct in comparison["improvements"]:
        status = "✨ Improved"
        change = f"{pct:+.1%}"
        current_str = format_value(metric, current)
        baseline_str = format_value(metric, baseline)
        lines.append(
            f"| {metric} | {current_str} | {baseline_str} | {change} | {status} |"
        )

    # Regressions (>threshold worse)
    for metric, current, baseline, pct in comparison["regressions"]:
        status = "⚠️ **REGRESSION**"
        change = f"{pct:+.1%}"
        current_str = format_value(metric, current)
        baseline_str = format_value(metric, baseline)
        lines.append(
            f"| {metric} | {current_str} | {baseline_str} | {change} | {status} |"
        )

    lines.append("")

    # Summary
    num_regressions = len(comparison["regressions"])
    num_improvements = len(comparison["improvements"])
    num_stable = len(comparison["stable"])

    if num_regressions > 0:
        lines.append(
            f"### ⚠️ {num_regressions} Regression(s) Detected"
        )
        lines.append("")
        lines.append(
            f"Performance has degraded by more than {threshold*100:.0f}% for {num_regressions} metric(s)."
        )
        lines.append(
            "Please investigate and optimize before merging, or update baselines if intentional."
        )
    else:
        lines.append("### ✅ No Regressions Detected")
        lines.append("")
        if num_improvements > 0:
            lines.append(
                f"Great work! {num_improvements} metric(s) improved by >5%! 🎉"
            )
        else:
            lines.append("All metrics are within acceptable thresholds.")

    lines.append("")
    lines.append("---")
    lines.append(
        f"_Threshold: ±{threshold*100:.0f}% | Platform: {platform} | [View History](../benchmarks/dashboard.html)_"
    )

    return "\n".join(lines)


def generate_html_report(
    run: Dict, comparison: Dict, platform: str, threshold: float
) -> str:
    """
    Generate HTML report with styling.

    Args:
        run: Latest performance run
        comparison: Comparison results
        platform: Platform identifier
        threshold: Regression threshold

    Returns:
        HTML-formatted report
    """
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Report - {platform}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }}
        .status-ok {{
            background: #4CAF50;
            color: white;
        }}
        .status-improved {{
            background: #2196F3;
            color: white;
        }}
        .status-regression {{
            background: #f44336;
            color: white;
        }}
        .metadata {{
            background: #f9f9f9;
            padding: 15px;
            border-left: 4px solid #4CAF50;
            margin: 20px 0;
        }}
        .summary {{
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
        }}
        .summary-ok {{
            background: #e8f5e9;
            border: 2px solid #4CAF50;
        }}
        .summary-warning {{
            background: #ffebee;
            border: 2px solid #f44336;
        }}
        .positive {{
            color: #4CAF50;
        }}
        .negative {{
            color: #f44336;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Performance Report</h1>

        <div class="metadata">
            <strong>Platform:</strong> {platform}<br>
            <strong>Timestamp:</strong> {run.get('timestamp', 'N/A')}<br>
"""

    if run.get("git_commit"):
        html += f"            <strong>Commit:</strong> {run['git_commit']}<br>\n"
    if run.get("git_branch"):
        html += f"            <strong>Branch:</strong> {run['git_branch']}<br>\n"

    html += "        </div>\n\n"

    # Check if we have metrics
    has_metrics = (
        comparison["regressions"] or comparison["improvements"] or comparison["stable"]
    )

    if not has_metrics:
        html += """
        <div class="summary summary-ok">
            <h2>⚠️ No Baseline Available</h2>
            <p>This appears to be the first run. Future runs will be compared against this baseline.</p>
        </div>
    </div>
</body>
</html>
"""
        return html

    # Metrics table
    html += "        <h2>📈 Metrics Comparison</h2>\n"
    html += "        <table>\n"
    html += "            <thead>\n"
    html += "                <tr>\n"
    html += "                    <th>Metric</th>\n"
    html += "                    <th>Current</th>\n"
    html += "                    <th>Baseline</th>\n"
    html += "                    <th>Change</th>\n"
    html += "                    <th>Status</th>\n"
    html += "                </tr>\n"
    html += "            </thead>\n"
    html += "            <tbody>\n"

    # All metrics
    all_metrics = (
        comparison["stable"] + comparison["improvements"] + comparison["regressions"]
    )

    for metric, current, baseline, pct in all_metrics:
        current_str = format_value(metric, current)
        baseline_str = format_value(metric, baseline)
        change_str = f"{pct:+.1%}"

        # Determine status and styling
        if (metric, current, baseline, pct) in comparison["regressions"]:
            status_class = "status-regression"
            status_text = "⚠️ REGRESSION"
            change_class = "negative"
        elif (metric, current, baseline, pct) in comparison["improvements"]:
            status_class = "status-improved"
            status_text = "✨ Improved"
            change_class = "positive"
        else:
            status_class = "status-ok"
            status_text = "✅ OK"
            change_class = ""

        html += f"""                <tr>
                    <td><strong>{metric}</strong></td>
                    <td>{current_str}</td>
                    <td>{baseline_str}</td>
                    <td class="{change_class}">{change_str}</td>
                    <td><span class="status {status_class}">{status_text}</span></td>
                </tr>
"""

    html += "            </tbody>\n"
    html += "        </table>\n\n"

    # Summary
    num_regressions = len(comparison["regressions"])
    num_improvements = len(comparison["improvements"])

    summary_class = "summary-warning" if num_regressions > 0 else "summary-ok"
    html += f'        <div class="summary {summary_class}">\n'

    if num_regressions > 0:
        html += f"            <h2>⚠️ {num_regressions} Regression(s) Detected</h2>\n"
        html += f"            <p>Performance has degraded by more than {threshold*100:.0f}% for {num_regressions} metric(s).</p>\n"
        html += "            <p>Please investigate and optimize before merging, or update baselines if intentional.</p>\n"
    else:
        html += "            <h2>✅ No Regressions Detected</h2>\n"
        if num_improvements > 0:
            html += f"            <p>Great work! {num_improvements} metric(s) improved by &gt;5%! 🎉</p>\n"
        else:
            html += "            <p>All metrics are within acceptable thresholds.</p>\n"

    html += "        </div>\n"

    html += """
    </div>
</body>
</html>
"""

    return html


def generate_json_report(
    run: Dict, comparison: Dict, platform: str, threshold: float
) -> str:
    """
    Generate JSON report for programmatic consumption.

    Args:
        run: Latest performance run
        comparison: Comparison results
        platform: Platform identifier
        threshold: Regression threshold

    Returns:
        JSON-formatted report
    """
    report = {
        "platform": platform,
        "timestamp": run.get("timestamp"),
        "git_commit": run.get("git_commit"),
        "git_branch": run.get("git_branch"),
        "threshold": threshold,
        "summary": {
            "num_regressions": len(comparison["regressions"]),
            "num_improvements": len(comparison["improvements"]),
            "num_stable": len(comparison["stable"]),
            "passed": len(comparison["regressions"]) == 0,
        },
        "metrics": {
            "regressions": [
                {
                    "metric": metric,
                    "current": current,
                    "baseline": baseline,
                    "change_pct": pct,
                }
                for metric, current, baseline, pct in comparison["regressions"]
            ],
            "improvements": [
                {
                    "metric": metric,
                    "current": current,
                    "baseline": baseline,
                    "change_pct": pct,
                }
                for metric, current, baseline, pct in comparison["improvements"]
            ],
            "stable": [
                {
                    "metric": metric,
                    "current": current,
                    "baseline": baseline,
                    "change_pct": pct,
                }
                for metric, current, baseline, pct in comparison["stable"]
            ],
        },
    }

    return json.dumps(report, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate performance report from latest run"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "json"],
        default="markdown",
        help="Report format (default: markdown)",
    )
    parser.add_argument(
        "--output", type=str, default="performance-report.md", help="Output file path"
    )
    parser.add_argument(
        "--platform",
        type=str,
        help="Platform identifier (default: auto-detect)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Regression threshold (default: from env or 0.20)",
    )

    args = parser.parse_args()

    # Detect platform
    if args.platform:
        platform = args.platform
    else:
        platform = detect_platform()
        log.info(f"Detected platform: {platform}")

    # Initialize history
    try:
        history = PerformanceHistory()
    except Exception as e:
        log.error(f"Failed to initialize performance history: {e}")
        sys.exit(1)

    # Get latest run
    recent_runs = history.get_recent_runs(platform, limit=1)

    if not recent_runs:
        log.warning(f"No performance runs found for platform: {platform}")
        # Create empty report
        if args.format == "markdown":
            report = f"## Performance Report\n\n⚠️ No performance data available for platform: `{platform}`\n\nRun performance tests to establish baseline.\n"
        elif args.format == "html":
            report = f"<html><body><h1>Performance Report</h1><p>No data available for {platform}</p></body></html>"
        else:
            report = json.dumps({"error": "No data available", "platform": platform})

        with open(args.output, "w") as f:
            f.write(report)

        log.info(f"Empty report written to {args.output}")
        return

    run = recent_runs[0]

    # Get current metrics from run
    current_metrics = {}
    for key, value in run.items():
        if key in [
            "embedding_throughput",
            "vector_search_latency",
            "query_latency_no_vllm",
            "query_latency_vllm",
            "db_insertion_throughput",
            "peak_memory_gb",
            "avg_memory_gb",
            "cache_hit_rate",
            "tokens_per_second",
            "avg_mrr",
            "avg_ndcg",
        ] and value is not None:
            current_metrics[key] = value

    # Compare to baseline
    threshold = args.threshold or float(
        __import__("os").getenv("REGRESSION_THRESHOLD", "0.20")
    )
    comparison = history.compare_to_baseline(current_metrics, platform, threshold)

    # Generate report
    if args.format == "markdown":
        report = generate_markdown_report(run, comparison, platform, threshold)
    elif args.format == "html":
        report = generate_html_report(run, comparison, platform, threshold)
    else:  # json
        report = generate_json_report(run, comparison, platform, threshold)

    # Write report
    with open(args.output, "w") as f:
        f.write(report)

    log.info(f"Report written to {args.output}")

    # Exit with error if regressions detected (for CI)
    if len(comparison["regressions"]) > 0:
        log.error(f"❌ {len(comparison['regressions'])} regression(s) detected")
        sys.exit(1)
    else:
        log.info("✅ No regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
