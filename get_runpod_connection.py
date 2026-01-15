#!/usr/bin/env python3
"""
Quick script to get PostgreSQL connection details from RunPod API.
Usage: python get_runpod_connection.py
"""

import os
from utils.runpod_manager import RunPodManager

# Get API key from environment
api_key = os.getenv("RUNPOD_API_KEY")
if not api_key:
    print("❌ Error: RUNPOD_API_KEY environment variable not set")
    print("   Set it with: export RUNPOD_API_KEY=your_key_here")
    exit(1)

print("🔍 Fetching RunPod pods...")
print()

manager = RunPodManager(api_key=api_key)

# List all pods
pods = manager.list_pods()

if not pods:
    print("❌ No pods found!")
    exit(1)

print(f"Found {len(pods)} pod(s):")
print()

# Show all pods with their connection details
for i, pod in enumerate(pods, 1):
    pod_id = pod.get('id', 'unknown')
    name = pod.get('name', 'unknown')
    runtime = pod.get('runtime') or {}
    machine = pod.get('machine') or {}

    status = runtime.get('containerState', 'unknown')
    uptime = runtime.get('uptimeInSeconds', 0)

    print(f"{'='*60}")
    print(f"Pod {i}: {name}")
    print(f"{'='*60}")
    print(f"ID:     {pod_id}")
    print(f"Status: {status}")
    print(f"Uptime: {uptime}s")
    print()

    # Get port mappings
    ports = runtime.get('ports', [])

    if ports:
        print("📡 Port Mappings:")
        for port_info in ports:
            private_port = port_info.get('privatePort')
            public_port = port_info.get('publicPort')
            ip = port_info.get('ip', 'N/A')
            port_type = port_info.get('type', 'tcp')

            # Highlight PostgreSQL port
            if private_port == 5432:
                print(f"   🐘 PostgreSQL:")
                print(f"      Internal: {private_port}/{port_type}")
                print(f"      External: {ip}:{public_port}")
                print()
                print(f"   ✅ Use these in Railway:")
                print(f"      PGHOST={ip}")
                print(f"      PGPORT={public_port}")
            else:
                print(f"   {private_port}/{port_type} → {ip}:{public_port}")
        print()
    else:
        print("⚠️  No port mappings found (pod may still be starting)")
        print()

print("="*60)
