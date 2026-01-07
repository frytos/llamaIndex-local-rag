#!/usr/bin/env python3
"""
Performance Dashboard Generator

Generate interactive Plotly dashboard showing performance trends over time.

Usage:
    python scripts/generate_performance_dashboard.py --days 30 --output dashboard.html
    python scripts/generate_performance_dashboard.py --platforms M1_Mac_16GB GitHub_Actions_macOS

Environment Variables:
    PERFORMANCE_DB_PATH: Path to performance database
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("ERROR: plotly not installed. Install with: pip install plotly")
    sys.exit(1)

from utils.performance_history import PerformanceHistory
from utils.platform_detection import detect_platform

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def get_available_platforms(history: PerformanceHistory) -> List[str]:
    """Get list of platforms with data."""
    return history.get_available_platforms()


def create_metric_subplot(
    fig,
    history: PerformanceHistory,
    platforms: List[str],
    metric: str,
    days: int,
    row: int,
    col: int,
    title: str,
    colors: List[str]
):
    """
    Add a metric subplot to the dashboard.

    Args:
        fig: Plotly figure object
        history: PerformanceHistory instance
        platforms: List of platforms to plot
        metric: Metric name
        days: Number of days of history
        row: Subplot row
        col: Subplot column
        title: Subplot title
        colors: List of colors for platforms
    """
    for idx, platform in enumerate(platforms):
        color = colors[idx % len(colors)]

        # Get trend data
        trend = history.get_trend(platform, metric, days)

        if not trend:
            continue

        timestamps, values = zip(*trend)

        # Add trace
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode='lines+markers',
                name=platform if row == 1 and col == 1 else None,
                line=dict(color=color, width=2),
                marker=dict(size=6),
                showlegend=(row == 1 and col == 1),  # Only show legend once
                hovertemplate=f"<b>{platform}</b><br>" +
                              "%{x|%Y-%m-%d %H:%M}<br>" +
                              f"{title}: %{{y:.3f}}<extra></extra>"
            ),
            row=row,
            col=col
        )

        # Get baseline
        baseline = history.get_baseline(platform, metric)

        if baseline:
            # Add baseline line
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color=color,
                opacity=0.5,
                annotation_text=f"Baseline: {baseline:.3f}",
                annotation_position="right",
                row=row,
                col=col
            )

            # Add regression threshold (20% above/below baseline)
            # For latency metrics: higher is worse
            # For throughput metrics: lower is worse
            if metric in ["vector_search_latency", "query_latency_no_vllm",
                         "query_latency_vllm", "peak_memory_gb", "avg_memory_gb"]:
                threshold = baseline * 1.2  # 20% worse (higher)
                threshold_text = "Regression (>20%)"
            else:
                threshold = baseline * 0.8  # 20% worse (lower)
                threshold_text = "Regression (<-20%)"

            fig.add_hline(
                y=threshold,
                line_dash="dot",
                line_color="red",
                opacity=0.3,
                annotation_text=threshold_text,
                annotation_position="left",
                row=row,
                col=col
            )

    # Update axis labels
    fig.update_yaxes(title_text=title, row=row, col=col)


def generate_dashboard(
    days: int = 30,
    platforms: Optional[List[str]] = None,
    output: str = "benchmarks/dashboard.html"
):
    """
    Generate interactive Plotly dashboard.

    Args:
        days: Number of days to show
        platforms: Platforms to include (None = all)
        output: Output HTML file path
    """
    log.info("Initializing performance dashboard...")

    # Initialize history
    history = PerformanceHistory()

    # Get platforms
    if platforms is None:
        platforms = get_available_platforms(history)

    if not platforms:
        log.warning("No platforms with data found")
        # Create empty dashboard
        fig = go.Figure()
        fig.add_annotation(
            text="No performance data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(title="RAG Pipeline Performance Dashboard")
        fig.write_html(output)
        log.info(f"Empty dashboard saved to {output}")
        return

    log.info(f"Generating dashboard for platforms: {', '.join(platforms)}")

    # Create subplots
    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=(
            "Query Latency (s)",
            "Embedding Throughput (chunks/s)",
            "Vector Search Latency (s)",
            "Memory Usage (GB)",
            "Cache Hit Rate (%)",
            "DB Insertion Throughput (nodes/s)",
            "RAG Quality (MRR)",
            "Tokens per Second"
        ),
        specs=[[{"secondary_y": False}] * 2] * 4,
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # Color palette
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f"]

    # Add metrics
    # Row 1
    create_metric_subplot(
        fig, history, platforms, "query_latency_no_vllm", days,
        1, 1, "Query Latency (s)", colors
    )

    create_metric_subplot(
        fig, history, platforms, "embedding_throughput", days,
        1, 2, "Embedding Throughput (c/s)", colors
    )

    # Row 2
    create_metric_subplot(
        fig, history, platforms, "vector_search_latency", days,
        2, 1, "Vector Search Latency (s)", colors
    )

    create_metric_subplot(
        fig, history, platforms, "peak_memory_gb", days,
        2, 2, "Peak Memory (GB)", colors
    )

    # Row 3
    create_metric_subplot(
        fig, history, platforms, "cache_hit_rate", days,
        3, 1, "Cache Hit Rate", colors
    )

    create_metric_subplot(
        fig, history, platforms, "db_insertion_throughput", days,
        3, 2, "DB Insertion (n/s)", colors
    )

    # Row 4
    create_metric_subplot(
        fig, history, platforms, "avg_mrr", days,
        4, 1, "MRR (Quality)", colors
    )

    create_metric_subplot(
        fig, history, platforms, "tokens_per_second", days,
        4, 2, "Tokens per Second", colors
    )

    # Update layout
    fig.update_layout(
        title={
            'text': "📊 RAG Pipeline Performance Dashboard",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24}
        },
        height=1600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified',
        template='plotly_white'
    )

    # Update all x-axes to show dates nicely
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # Save to file
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path))
    log.info(f"✅ Dashboard saved to {output}")

    # Print summary
    stats = history.get_stats()
    log.info(f"\nDashboard Statistics:")
    log.info(f"  Total runs: {stats['total_runs']}")
    log.info(f"  Platforms: {stats['num_platforms']}")
    log.info(f"  Date range: {stats['first_run']} to {stats['last_run']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate interactive performance dashboard"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to show (default: 30)"
    )
    parser.add_argument(
        "--platforms",
        nargs="+",
        help="Platforms to include (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/dashboard.html",
        help="Output HTML file (default: benchmarks/dashboard.html)"
    )

    args = parser.parse_args()

    try:
        generate_dashboard(
            days=args.days,
            platforms=args.platforms,
            output=args.output
        )
    except Exception as e:
        log.error(f"Failed to generate dashboard: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
