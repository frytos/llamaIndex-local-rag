#!/usr/bin/env python3
"""
Analyze and visualize resource consumption from query comparison runs.
"""
import re
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def parse_resource_log(filepath):
    """Parse a resource log file and extract swap usage."""
    with open(filepath, 'r') as f:
        content = f.read()

    data = {
        'baseline_swap_total': 0,
        'baseline_swap_used': 0,
        'peak_swap_total': 0,
        'peak_swap_used': 0,
    }

    # Parse baseline
    baseline_match = re.search(r'=== BASELINE @ (\d+:\d+:\d+) ===.*?Swap: vm\.swapusage: total = ([\d.]+)M\s+used = ([\d.]+)M', content, re.DOTALL)
    if baseline_match:
        data['baseline_time'] = baseline_match.group(1)
        data['baseline_swap_total'] = float(baseline_match.group(2))
        data['baseline_swap_used'] = float(baseline_match.group(3))

    # Parse peak
    peak_match = re.search(r'=== PEAK @ (\d+:\d+:\d+) ===.*?Swap: vm\.swapusage: total = ([\d.]+)M\s+used = ([\d.]+)M', content, re.DOTALL)
    if peak_match:
        data['peak_time'] = peak_match.group(1)
        data['peak_swap_total'] = float(peak_match.group(2))
        data['peak_swap_used'] = float(peak_match.group(3))

    return data

def parse_timing_log(filepath):
    """Parse a timing log file."""
    with open(filepath, 'r') as f:
        content = f.read()

    data = {}

    # Parse total time
    total_match = re.search(r'Total: ([\d.]+)s', content)
    if total_match:
        data['total_time'] = float(total_match.group(1))

    # Parse model
    model_match = re.search(r'Model: (\w+)', content)
    if model_match:
        data['model'] = model_match.group(1)

    # Parse query number
    query_match = re.search(r'Query: (\d+)', content)
    if query_match:
        data['query_num'] = int(query_match.group(1))

    return data

def collect_data(run_dir):
    """Collect all resource and timing data from a run directory."""
    run_path = Path(run_dir)
    data = []

    # Find all resource logs
    for resource_file in sorted(run_path.glob('q*_*_resources.log')):
        # Extract query number and model from filename
        match = re.match(r'q(\d+)_(\w+)_resources\.log', resource_file.name)
        if not match:
            continue

        query_num = int(match.group(1))
        model = match.group(2)

        # Parse resource log
        resource_data = parse_resource_log(resource_file)

        # Find corresponding timing log
        timing_file = run_path / f'q{query_num}_{model}_timing.txt'
        timing_data = {}
        if timing_file.exists():
            timing_data = parse_timing_log(timing_file)

        # Combine data
        entry = {
            'run': run_path.name,
            'query_num': query_num,
            'model': model,
            **resource_data,
            **timing_data
        }
        data.append(entry)

    return data

def create_visualizations(run1_dir, run2_dir):
    """Create comprehensive resource usage visualizations."""

    # Collect data
    print("📊 Collecting data from runs...")
    run1_data = collect_data(run1_dir)
    run2_data = collect_data(run2_dir)

    print(f"  Run 1: {len(run1_data)} queries")
    print(f"  Run 2: {len(run2_data)} queries")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # ============================================================
    # 1. Swap Usage Comparison (Baseline vs Peak)
    # ============================================================
    ax1 = plt.subplot(2, 3, 1)

    run1_queries = [d['query_num'] for d in run1_data if d['model'] == 'bge']
    run1_baseline = [d['baseline_swap_used'] for d in run1_data if d['model'] == 'bge']
    run1_peak = [d['peak_swap_used'] for d in run1_data if d['model'] == 'bge']

    run2_queries = [d['query_num'] for d in run2_data if d['model'] == 'bge']
    run2_baseline = [d['baseline_swap_used'] for d in run2_data if d['model'] == 'bge']
    run2_peak = [d['peak_swap_used'] for d in run2_data if d['model'] == 'bge']

    x1 = np.arange(len(run1_queries))
    x2 = np.arange(len(run2_queries))
    width = 0.35

    ax1.bar(x1 - width/2, run1_baseline, width, label='Run1 Baseline', color='#3498db', alpha=0.7)
    ax1.bar(x1 + width/2, run1_peak, width, label='Run1 Peak', color='#e74c3c', alpha=0.7)

    ax1.set_xlabel('Query Number (BGE)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Swap Usage (MB)', fontsize=11, fontweight='bold')
    ax1.set_title('Run 1: Swap Usage - Baseline vs Peak\n(BGE Model)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(run1_queries)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='No Swap = Optimal')

    # ============================================================
    # 2. Run 2 Swap Usage (showing the problem!)
    # ============================================================
    ax2 = plt.subplot(2, 3, 2)

    ax2.bar(x2 - width/2, run2_baseline, width, label='Run2 Baseline', color='#f39c12', alpha=0.7)
    ax2.bar(x2 + width/2, run2_peak, width, label='Run2 Peak', color='#c0392b', alpha=0.7)

    ax2.set_xlabel('Query Number (BGE)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Swap Usage (MB)', fontsize=11, fontweight='bold')
    ax2.set_title('Run 2: Swap Usage - Baseline vs Peak\n⚠️ SWAP ACTIVE! (BGE Model)', fontsize=13, fontweight='bold', color='red')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(run2_queries)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=1000, color='orange', linestyle='--', linewidth=2, label='1GB Swap')
    ax2.axhline(y=2000, color='red', linestyle='--', linewidth=2, label='2GB Swap (Critical)')

    # ============================================================
    # 3. Swap Delta (Peak - Baseline) per Query
    # ============================================================
    ax3 = plt.subplot(2, 3, 3)

    run1_delta = [peak - baseline for peak, baseline in zip(run1_peak, run1_baseline)]
    run2_delta = [peak - baseline for peak, baseline in zip(run2_peak, run2_baseline)]

    x_all = np.arange(max(len(run1_queries), len(run2_queries)))

    ax3.plot(run1_queries, run1_delta, 'o-', color='#3498db', linewidth=2, markersize=8, label='Run 1 (No swap pressure)')
    ax3.plot(run2_queries, run2_delta, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='Run 2 (Swap active)')

    ax3.set_xlabel('Query Number', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Swap Increase (MB)', fontsize=11, fontweight='bold')
    ax3.set_title('Swap Usage Increase During Query\n(Peak - Baseline)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=1)

    # ============================================================
    # 4. Query Time vs Swap Usage Correlation
    # ============================================================
    ax4 = plt.subplot(2, 3, 4)

    run1_times = [d.get('total_time', 0) for d in run1_data if d['model'] == 'bge']
    run2_times = [d.get('total_time', 0) for d in run2_data if d['model'] == 'bge']

    # Scatter plot
    ax4.scatter(run1_baseline, run1_times, s=100, alpha=0.6, color='#3498db', label='Run 1', edgecolors='black')
    ax4.scatter(run2_baseline, run2_times, s=100, alpha=0.6, color='#e74c3c', label='Run 2', edgecolors='black')

    # Add trend line for Run 2
    if len(run2_baseline) > 1 and len(run2_times) > 1:
        z = np.polyfit(run2_baseline, run2_times, 1)
        p = np.poly1d(z)
        ax4.plot(run2_baseline, p(run2_baseline), "r--", alpha=0.8, linewidth=2, label=f'Run 2 Trend')

    ax4.set_xlabel('Baseline Swap Usage (MB)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Query Time (seconds)', fontsize=11, fontweight='bold')
    ax4.set_title('Query Time vs Swap Usage\n(BGE Model)', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # ============================================================
    # 5. BGE vs E5 Swap Comparison (Run 2)
    # ============================================================
    ax5 = plt.subplot(2, 3, 5)

    run2_bge = [d for d in run2_data if d['model'] == 'bge']
    run2_e5 = [d for d in run2_data if d['model'] == 'e5']

    bge_queries = [d['query_num'] for d in run2_bge]
    bge_peak = [d['peak_swap_used'] for d in run2_bge]

    e5_queries = [d['query_num'] for d in run2_e5]
    e5_peak = [d['peak_swap_used'] for d in run2_e5]

    x_bge = np.arange(len(bge_queries))
    width = 0.35

    ax5.bar(x_bge - width/2, bge_peak, width, label='BGE', color='#9b59b6', alpha=0.7)
    ax5.bar(x_bge + width/2, e5_peak, width, label='E5', color='#1abc9c', alpha=0.7)

    ax5.set_xlabel('Query Number', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Peak Swap Usage (MB)', fontsize=11, fontweight='bold')
    ax5.set_title('Run 2: BGE vs E5 Peak Swap Usage', fontsize=13, fontweight='bold')
    ax5.set_xticks(x_bge)
    ax5.set_xticklabels(bge_queries)
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # ============================================================
    # 6. Summary Statistics Table
    # ============================================================
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    # Calculate statistics
    run1_avg_baseline = np.mean(run1_baseline) if run1_baseline else 0
    run1_avg_peak = np.mean(run1_peak) if run1_peak else 0
    run1_avg_time = np.mean(run1_times) if run1_times else 0

    run2_avg_baseline = np.mean(run2_baseline) if run2_baseline else 0
    run2_avg_peak = np.mean(run2_peak) if run2_peak else 0
    run2_avg_time = np.mean(run2_times) if run2_times else 0

    stats_text = f"""
    📊 RESOURCE CONSUMPTION SUMMARY

    Run 1 (04:19) - Normal Operation:
      • Avg Baseline Swap: {run1_avg_baseline:.1f} MB
      • Avg Peak Swap: {run1_avg_peak:.1f} MB
      • Avg Query Time: {run1_avg_time:.1f}s
      • Status: ✅ NO SWAP PRESSURE

    Run 2 (04:57) - Degraded Performance:
      • Avg Baseline Swap: {run2_avg_baseline:.1f} MB
      • Avg Peak Swap: {run2_avg_peak:.1f} MB
      • Avg Query Time: {run2_avg_time:.1f}s
      • Status: 🔴 HIGH SWAP USAGE

    🔍 Key Findings:
      • Run 2 started with {run2_avg_baseline:.0f}MB swap active
      • Swap increased to {run2_avg_peak:.0f}MB during queries
      • Query time increased {run2_avg_time/run1_avg_time:.1f}x vs Run 1
      • Correlation: High swap → Slow queries

    💡 Recommendation:
      • Free system memory before benchmarking
      • Monitor swap usage in real-time
      • Restart system if swap > 500MB
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ============================================================
    # Final adjustments
    # ============================================================
    plt.suptitle('🔬 RAG Pipeline Resource Consumption Analysis\nComparison: Run 1 (04:19) vs Run 2 (04:57)',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save figure
    output_file = 'resource_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_file}")

    # Show plot
    plt.show()

if __name__ == '__main__':
    run1_dir = '/Users/frytos/code/llamaIndex-local-rag/query_comparison_20251220_041917'
    run2_dir = '/Users/frytos/code/llamaIndex-local-rag/query_comparison_20251220_045713'

    create_visualizations(run1_dir, run2_dir)
