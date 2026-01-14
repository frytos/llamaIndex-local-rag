#!/usr/bin/env python3
"""
List available RunPod GPU types and their IDs.

Usage:
    python scripts/list_runpod_gpus.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import RunPodManager

def main():
    # Get API key from environment
    api_key = os.getenv('RUNPOD_API_KEY')

    if not api_key:
        print("❌ Error: RUNPOD_API_KEY not set")
        print("   Set it with: export RUNPOD_API_KEY=your_key_here")
        print("   Get your key from: https://runpod.io/settings")
        sys.exit(1)

    print("=" * 80)
    print("AVAILABLE RUNPOD GPU TYPES")
    print("=" * 80)
    print()

    manager = RunPodManager(api_key=api_key)
    gpus = manager.list_available_gpus()

    if not gpus:
        print("❌ No GPUs found or API error")
        sys.exit(1)

    print(f"Found {len(gpus)} GPU types:\n")

    # Sort by price (low to high)
    gpus_sorted = sorted(
        gpus,
        key=lambda x: x.get('lowestPrice', {}).get('uninterruptablePrice', 999),
    )

    print(f"{'GPU ID':<25} {'Display Name':<30} {'VRAM':<10} {'Price/hr':<12} {'Available':<10}")
    print("-" * 95)

    for gpu in gpus_sorted:
        gpu_id = gpu.get('id', 'unknown')
        display_name = gpu.get('displayName', 'unknown')
        memory_mb = gpu.get('memoryInGb', 0)

        lowest_price = gpu.get('lowestPrice', {})
        price = lowest_price.get('uninterruptablePrice', 0) if lowest_price else 0

        # Check availability (has community instances)
        community_price = gpu.get('communityPrice', 0)
        available = "✅ Yes" if community_price and community_price > 0 else "❌ No"

        print(f"{gpu_id:<25} {display_name:<30} {memory_mb}GB{'':<5} ${price:<11.2f} {available:<10}")

    print()
    print("=" * 80)
    print("USAGE IN CODE:")
    print("=" * 80)
    print()
    print("Use the 'GPU ID' column when creating pods:")
    print()
    print("  pod = manager.create_pod(")
    print("      name='my-pod',")
    print("      gpu_type='NVIDIA GeForce RTX 4090',  # ← Use exact ID from table")
    print("      volume_gb=100")
    print("  )")
    print()

if __name__ == "__main__":
    main()
