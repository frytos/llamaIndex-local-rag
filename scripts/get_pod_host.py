#!/usr/bin/env python3
"""
Get pod host from pod ID.

Usage:
    python scripts/get_pod_host.py POD_ID

Example:
    python scripts/get_pod_host.py 9b7ancer5y9ydm
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import RunPodManager

if len(sys.argv) < 2:
    print("Usage: python scripts/get_pod_host.py POD_ID")
    sys.exit(1)

pod_id = sys.argv[1]

api_key = os.getenv('RUNPOD_API_KEY')
if not api_key:
    print("❌ Error: RUNPOD_API_KEY not set")
    print("   Set it with: export RUNPOD_API_KEY=your_key_here")
    sys.exit(1)

manager = RunPodManager(api_key=api_key)
pod = manager.get_pod(pod_id)

if not pod:
    print(f"❌ Pod {pod_id} not found")
    sys.exit(1)

machine = pod.get('machine') or {}
pod_host = machine.get('podHostId', '')

if pod_host:
    print(f"✅ Pod Host: {pod_host}")
    print()
    print("Run automated setup:")
    print(f"  bash scripts/setup_runpod_pod.sh {pod_host}")
else:
    print(f"⚠️  Pod host not available yet for pod {pod_id}")
    print("   Wait a moment and try again")
