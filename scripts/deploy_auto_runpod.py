#!/usr/bin/env python3
"""
Fully Automated RunPod Deployment with GitHub Auto-Setup

Creates a pod that automatically:
1. Clones your repository from GitHub
2. Installs PostgreSQL + pgvector
3. Sets up Python environment
4. Starts vLLM server
5. Everything ready when you SSH in!

Usage:
    python scripts/deploy_auto_runpod.py --github https://github.com/user/repo.git

    Or:
    python scripts/deploy_auto_runpod.py --github https://github.com/user/repo.git --branch dev --name my-pod
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import create_auto_pod

def main():
    parser = argparse.ArgumentParser(
        description='Fully automated RunPod deployment with GitHub auto-setup'
    )
    parser.add_argument(
        '--github',
        required=True,
        help='GitHub repository URL (e.g., https://github.com/user/llamaIndex-local-rag.git)'
    )
    parser.add_argument(
        '--branch',
        default='main',
        help='Git branch to clone (default: main)'
    )
    parser.add_argument(
        '--name',
        help='Pod name (default: auto-generated)'
    )
    parser.add_argument(
        '--gpu',
        default='NVIDIA GeForce RTX 4090',
        help='GPU type (default: RTX 4090)'
    )
    parser.add_argument(
        '--volume',
        type=int,
        default=100,
        help='Volume size in GB (default: 100)'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Do not wait for pod to be ready'
    )

    args = parser.parse_args()

    # Generate name if not provided
    if not args.name:
        import time
        args.name = f"rag-auto-{int(time.time())}"

    print("=" * 80)
    print("AUTOMATED RUNPOD DEPLOYMENT")
    print("=" * 80)
    print(f"Repository: {args.github}")
    print(f"Branch: {args.branch}")
    print(f"Pod Name: {args.name}")
    print(f"GPU: {args.gpu}")
    print(f"Storage: {args.volume}GB")
    print("=" * 80)
    print()

    pod = create_auto_pod(
        name=args.name,
        github_repo=args.github,
        github_branch=args.branch,
        gpu_type=args.gpu,
        volume_gb=args.volume,
        wait=not args.no_wait
    )

    if not pod:
        print("❌ Deployment failed")
        sys.exit(1)

    machine = pod.get('machine') or {}
    ssh_host = machine.get('podHostId', 'unknown')

    print()
    print("=" * 80)
    print("🎉 POD DEPLOYED!")
    print("=" * 80)
    print()
    print(f"Pod ID: {pod['id']}")
    print(f"Name: {args.name}")
    print(f"SSH Host: {ssh_host}")
    print()
    print("Services are initializing in background (~10-15 minutes)...")
    print()
    print("Connect and check progress:")
    print(f"  ssh {ssh_host}@ssh.runpod.io")
    print(f"  tail -f /workspace/startup.log")
    print()
    print("When complete, create SSH tunnel:")
    print(f"  ssh -N -L 8000:localhost:8000 -L 5432:localhost:5432 {ssh_host}@ssh.runpod.io")
    print()
    print("Test services:")
    print("  curl http://localhost:8000/health")
    print("  psql -h localhost -U fryt -d vector_db -c 'SELECT 1'")
    print()

if __name__ == "__main__":
    main()
