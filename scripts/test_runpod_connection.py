#!/usr/bin/env python3
"""
Test RunPod API Connection

Validates RunPod API key and tests basic operations.

Usage:
    python scripts/test_runpod_connection.py --api-key YOUR_KEY
    python scripts/test_runpod_connection.py  # Uses RUNPOD_API_KEY env var
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import RunPodManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def test_connection(api_key: str):
    """Test RunPod API connection and basic operations."""

    print("=" * 70)
    print("RUNPOD API CONNECTION TEST")
    print("=" * 70)
    print()

    # Step 1: Initialize manager
    print("📝 Step 1: Initializing RunPod manager...")
    try:
        manager = RunPodManager(api_key=api_key)
        print("✅ Manager initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize manager: {e}")
        return False

    print()

    # Step 2: Test API connection by listing pods
    print("📝 Step 2: Testing API connection (list pods)...")
    try:
        pods = manager.list_pods()
        print(f"✅ API connection successful!")
        print(f"   Found {len(pods)} existing pods")

        if pods:
            print("\n   Existing pods:")
            for pod in pods[:5]:  # Show first 5
                name = pod.get('name', 'N/A')
                runtime = pod.get('runtime', {})
                status = runtime.get('containerState', 'unknown')
                machine = pod.get('machine', {})
                gpu = machine.get('gpuTypeId', 'N/A')

                print(f"     • {name}: {status} ({gpu})")

            if len(pods) > 5:
                print(f"     ... and {len(pods) - 5} more")
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

    print()

    # Step 3: List available GPUs
    print("📝 Step 3: Checking available GPU types...")
    try:
        gpus = manager.list_available_gpus()

        if gpus:
            print(f"✅ Found {len(gpus)} GPU types available")
            print("\n   Recommended GPUs for RAG:")

            recommended = [
                "NVIDIA RTX 4090",
                "NVIDIA RTX 4070 Ti",
                "NVIDIA RTX 3090"
            ]

            for gpu in gpus:
                name = gpu.get('displayName', 'N/A')
                if any(rec in name for rec in recommended):
                    memory = gpu.get('memoryInGb', 0)
                    lowest_price = gpu.get('lowestPrice', {})
                    price = lowest_price.get('uninterruptablePrice', 0)

                    print(f"     • {name}: {memory}GB VRAM, ${price:.2f}/hour")
        else:
            print("⚠️  No GPU types returned (API may have changed)")

    except Exception as e:
        print(f"⚠️  Failed to list GPUs: {e}")
        print("   (This is not critical - pod creation may still work)")

    print()

    # Step 4: Test cost estimation
    print("📝 Step 4: Testing cost estimation...")
    try:
        costs = manager.estimate_cost(hours_per_day=8, cost_per_hour=0.50)

        print("✅ Cost estimation working")
        print(f"\n   Example: RTX 4090 usage")
        print(f"     • 8 hours/day")
        print(f"     • ${costs['cost_per_hour']:.2f}/hour")
        print(f"     • Daily cost: ${costs['daily_cost']:.2f}")
        print(f"     • Monthly cost: ${costs['total_cost']:.2f}")

    except Exception as e:
        print(f"⚠️  Cost estimation failed: {e}")

    print()

    # Step 5: Test SSH command generation (with mock pod)
    print("📝 Step 5: Testing utility functions...")
    try:
        # This will show "not found" but tests the function logic
        ssh_cmd = manager.get_ssh_command("test-pod-id")
        print("✅ SSH command generator working")
        print(f"   Example: {ssh_cmd[:80]}...")

    except Exception as e:
        print(f"⚠️  Utility function test failed: {e}")

    print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ All critical tests passed!")
    print()
    print("Your RunPod API key is valid and working.")
    print("You can now use the RunPodManager to:")
    print("  • Create pods")
    print("  • Manage pod lifecycle")
    print("  • Monitor status and metrics")
    print()
    print("Next steps:")
    print("  1. Review config/runpod_deployment.env")
    print("  2. Create a pod: python scripts/deploy_to_runpod.py")
    print("  3. Check docs/RUNPOD_DEPLOYMENT_WORKFLOW.md")
    print()

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Test RunPod API connection'
    )
    parser.add_argument(
        '--api-key',
        help='RunPod API key (or set RUNPOD_API_KEY env var)'
    )

    args = parser.parse_args()

    api_key = args.api_key

    if not api_key:
        import os
        api_key = os.getenv('RUNPOD_API_KEY')

    if not api_key:
        print("❌ Error: No API key provided")
        print()
        print("Usage:")
        print("  1. Pass as argument:")
        print("     python scripts/test_runpod_connection.py --api-key YOUR_KEY")
        print()
        print("  2. Set environment variable:")
        print("     export RUNPOD_API_KEY=your_key")
        print("     python scripts/test_runpod_connection.py")
        print()
        print("Get your API key from: https://runpod.io/settings")
        sys.exit(1)

    # Run tests
    success = test_connection(api_key)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
