#!/usr/bin/env python3
"""
Get SSH connection info for RunPod pods.

Usage:
    python scripts/get_pod_ssh.py
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
        sys.exit(1)

    print("=" * 80)
    print("RUNPOD PODS - SSH CONNECTION INFO")
    print("=" * 80)
    print()

    manager = RunPodManager(api_key=api_key)
    pods = manager.list_pods()

    if not pods:
        print("❌ No pods found")
        sys.exit(1)

    print(f"Found {len(pods)} pod(s):\n")

    for i, pod in enumerate(pods, 1):
        name = pod.get('name', 'unknown')
        pod_id = pod.get('id', 'unknown')

        runtime = pod.get('runtime') or {}
        machine = pod.get('machine') or {}

        state = runtime.get('containerState', 'unknown')
        ssh_host = machine.get('podHostId', 'N/A')

        print(f"{i}. {name}")
        print(f"   ID: {pod_id}")
        print(f"   Status: {state}")
        print(f"   SSH Host: {ssh_host}")
        print()

        if state == 'running' and ssh_host != 'N/A':
            print(f"   📋 SSH Command:")
            print(f"      ssh {ssh_host}@ssh.runpod.io")
            print()
            print(f"   📋 SSH Tunnel Command:")
            print(f"      ssh -N -L 8000:localhost:8000 -L 5432:localhost:5432 {ssh_host}@ssh.runpod.io")
            print()
        elif state != 'running':
            print(f"   ⚠️  Pod is not running (state: {state})")
            print()

        print("-" * 80)
        print()

    print("💡 Copy and paste the command above to connect!")
    print()

if __name__ == "__main__":
    main()
