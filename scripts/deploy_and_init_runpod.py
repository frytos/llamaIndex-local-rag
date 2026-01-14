#!/usr/bin/env python3
"""
Automated RunPod Deployment with Auto-Initialization

Creates a pod, uploads the init script, and runs it automatically.

Usage:
    python scripts/deploy_and_init_runpod.py

    Or with custom settings:
    python scripts/deploy_and_init_runpod.py --name my-pod --wait
"""

import argparse
import sys
import time
import os
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import RunPodManager

def main():
    parser = argparse.ArgumentParser(description='Deploy and auto-initialize RunPod')
    parser.add_argument('--api-key', help='RunPod API key (or set RUNPOD_API_KEY env var)')
    parser.add_argument('--name', default=f"rag-pipeline-{int(time.time())}", help='Pod name')
    parser.add_argument('--gpu', default='NVIDIA GeForce RTX 4090', help='GPU type')
    parser.add_argument('--volume', type=int, default=100, help='Volume size in GB')
    args = parser.parse_args()

    api_key = args.api_key or os.getenv('RUNPOD_API_KEY')

    print("Creating pod with auto-init...")
    # Implementation here
