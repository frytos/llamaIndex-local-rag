#!/usr/bin/env python3
"""
RunPod CLI Utility

Quick command-line interface for common RunPod operations.

Usage:
    # List all pods
    python scripts/runpod_cli.py list

    # Create pod
    python scripts/runpod_cli.py create --name my-pod

    # Stop pod
    python scripts/runpod_cli.py stop POD_ID

    # Resume pod
    python scripts/runpod_cli.py resume POD_ID

    # Get status
    python scripts/runpod_cli.py status POD_ID

    # SSH command
    python scripts/runpod_cli.py ssh POD_ID

    # Create tunnel
    python scripts/runpod_cli.py tunnel POD_ID

    # Cost estimate
    python scripts/runpod_cli.py cost 8
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runpod_manager import RunPodManager
from utils.ssh_tunnel import SSHTunnelManager
import os


def cmd_list(manager: RunPodManager, args):
    """List all pods."""
    pods = manager.list_pods()

    if not pods:
        print("No pods found")
        return

    print(f"\nFound {len(pods)} pods:\n")
    print(f"{'Name':<30} {'Status':<12} {'GPU':<20} {'Cost/hr'}")
    print("-" * 80)

    for pod in pods:
        name = pod.get('name', 'N/A')
        runtime = pod.get('runtime', {})
        status = runtime.get('containerState', 'unknown')
        machine = pod.get('machine', {})
        gpu = machine.get('gpuTypeId', 'N/A')
        cost = pod.get('costPerHr', 0)

        print(f"{name:<30} {status:<12} {gpu:<20} ${cost:.2f}")

    print()


def cmd_create(manager: RunPodManager, args):
    """Create new pod."""
    print(f"Creating pod: {args.name}")
    print(f"GPU: {args.gpu}")
    print(f"Volume: {args.volume}GB")
    print()

    pod = manager.create_pod(
        name=args.name,
        gpu_type=args.gpu,
        volume_gb=args.volume
    )

    if pod:
        print("\n✅ Pod created successfully!")
        print(f"   ID: {pod['id']}")
        print(f"   SSH: ssh {pod['machine']['podHostId']}@ssh.runpod.io")
        print()

        if args.wait:
            print("Waiting for pod to be ready...")
            if manager.wait_for_ready(pod['id']):
                print("✅ Pod is ready!")
            else:
                print("⚠️  Pod not ready yet. Check status manually.")
    else:
        print("\n❌ Failed to create pod")
        return 1


def cmd_stop(manager: RunPodManager, args):
    """Stop pod."""
    pod_id = args.pod_id

    print(f"Stopping pod: {pod_id}")

    if manager.stop_pod(pod_id):
        print("✅ Pod stopped successfully")
        print("   No more GPU costs (storage costs still apply)")
    else:
        print("❌ Failed to stop pod")
        return 1


def cmd_resume(manager: RunPodManager, args):
    """Resume pod."""
    pod_id = args.pod_id

    print(f"Resuming pod: {pod_id}")

    if manager.resume_pod(pod_id):
        print("✅ Pod resumed successfully")

        if args.wait:
            print("Waiting for pod to be ready...")
            if manager.wait_for_ready(pod_id):
                print("✅ Pod is ready!")
    else:
        print("❌ Failed to resume pod")
        return 1


def cmd_terminate(manager: RunPodManager, args):
    """Terminate pod."""
    pod_id = args.pod_id

    if not args.yes:
        confirm = input(f"⚠️  Terminate pod {pod_id}? This cannot be undone. [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("Cancelled")
            return 0

    print(f"Terminating pod: {pod_id}")

    if manager.terminate_pod(pod_id):
        print("✅ Pod terminated successfully")
        print("   All pod data has been deleted")
    else:
        print("❌ Failed to terminate pod")
        return 1


def cmd_status(manager: RunPodManager, args):
    """Get pod status."""
    pod_id = args.pod_id

    status = manager.get_pod_status(pod_id)

    if status.get('status') == 'not_found':
        print(f"❌ Pod {pod_id} not found")
        return 1

    print(f"\nPod Status: {pod_id}\n")
    print(f"  State: {status['status']}")
    print(f"  Uptime: {status['uptime_seconds']}s ({status['uptime_seconds'] // 60}min)")
    print(f"  GPU: {status['gpu_type']} ({status['gpu_count']}x)")
    print(f"  GPU Usage: {status['gpu_utilization']}%")
    print(f"  Memory Usage: {status['memory_utilization']}%")
    print(f"  Cost: ${status['cost_per_hour']:.2f}/hour")
    print(f"  SSH: {status['ssh_host']}@ssh.runpod.io:{status['ssh_port']}")
    print()


def cmd_ssh(manager: RunPodManager, args):
    """Get SSH command."""
    pod_id = args.pod_id

    ssh_cmd = manager.get_ssh_command(pod_id, ports=args.ports)
    print(f"\nSSH Command:\n")
    print(f"  {ssh_cmd}")
    print()

    if args.copy:
        try:
            import pyperclip
            pyperclip.copy(ssh_cmd)
            print("✅ Command copied to clipboard")
        except ImportError:
            print("⚠️  pyperclip not installed (cannot copy to clipboard)")


def cmd_tunnel(manager: RunPodManager, args):
    """Create SSH tunnel."""
    pod_id = args.pod_id

    # Get SSH host
    status = manager.get_pod_status(pod_id)

    if status.get('status') == 'not_found':
        print(f"❌ Pod {pod_id} not found")
        return 1

    ssh_host = status['ssh_host']

    print(f"Creating SSH tunnel to pod {pod_id}")
    print(f"Forwarding ports: {', '.join(map(str, args.ports))}")
    print()

    # Create tunnel
    tunnel = SSHTunnelManager(ssh_host)

    if tunnel.create_tunnel(ports=args.ports, background=args.background):
        print("✅ SSH tunnel created")
        print()
        print("Services available at:")
        for port in args.ports:
            service = {8000: "vLLM", 5432: "PostgreSQL", 3000: "Grafana"}.get(port, "Service")
            print(f"  {service}: localhost:{port}")
        print()

        if args.background:
            print(f"Tunnel running in background (PID: {tunnel.process.pid})")
            print("To stop: kill <PID>")
        else:
            print("Press Ctrl+C to stop tunnel")
            try:
                tunnel.process.wait()
            except KeyboardInterrupt:
                print("\nStopping tunnel...")
                tunnel.stop_tunnel()
    else:
        print("❌ Failed to create tunnel")
        return 1


def cmd_cost(manager: RunPodManager, args):
    """Estimate costs."""
    hours = args.hours_per_day

    costs = manager.estimate_cost(
        hours_per_day=hours,
        days=30,
        cost_per_hour=args.cost_per_hour
    )

    print(f"\nCost Estimation for RTX 4090:\n")
    print(f"  Usage: {hours} hours/day")
    print(f"  Rate: ${args.cost_per_hour:.2f}/hour")
    print(f"  Daily: ${costs['daily_cost']:.2f}")
    print(f"  Monthly (30 days): ${costs['total_cost']:.2f}")
    print()

    # Show scenarios
    print("Other scenarios:")
    for h in [2, 4, 8, 24]:
        c = manager.estimate_cost(h, cost_per_hour=args.cost_per_hour)
        print(f"  {h:2d} hours/day: ${c['total_cost']:6.2f}/month")
    print()


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='RunPod CLI Utility',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Global arguments
    parser.add_argument(
        '--api-key',
        help='RunPod API key (or set RUNPOD_API_KEY env var)'
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # List command
    subparsers.add_parser('list', help='List all pods')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create new pod')
    create_parser.add_argument('--name', default='rag-pipeline-vllm', help='Pod name')
    create_parser.add_argument('--gpu', default='NVIDIA RTX 4090', help='GPU type')
    create_parser.add_argument('--volume', type=int, default=100, help='Volume size (GB)')
    create_parser.add_argument('--wait', action='store_true', help='Wait for pod to be ready')

    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop pod')
    stop_parser.add_argument('pod_id', help='Pod ID')

    # Resume command
    resume_parser = subparsers.add_parser('resume', help='Resume pod')
    resume_parser.add_argument('pod_id', help='Pod ID')
    resume_parser.add_argument('--wait', action='store_true', help='Wait for pod to be ready')

    # Terminate command
    terminate_parser = subparsers.add_parser('terminate', help='Terminate pod (permanent)')
    terminate_parser.add_argument('pod_id', help='Pod ID')
    terminate_parser.add_argument('--yes', action='store_true', help='Skip confirmation')

    # Status command
    status_parser = subparsers.add_parser('status', help='Get pod status')
    status_parser.add_argument('pod_id', help='Pod ID')

    # SSH command
    ssh_parser = subparsers.add_parser('ssh', help='Get SSH command')
    ssh_parser.add_argument('pod_id', help='Pod ID')
    ssh_parser.add_argument('--ports', type=int, nargs='+', default=[8000, 5432, 3000], help='Ports to forward')
    ssh_parser.add_argument('--copy', action='store_true', help='Copy to clipboard')

    # Tunnel command
    tunnel_parser = subparsers.add_parser('tunnel', help='Create SSH tunnel')
    tunnel_parser.add_argument('pod_id', help='Pod ID')
    tunnel_parser.add_argument('--ports', type=int, nargs='+', default=[8000, 5432, 3000], help='Ports to forward')
    tunnel_parser.add_argument('--background', action='store_true', help='Run in background')

    # Cost command
    cost_parser = subparsers.add_parser('cost', help='Estimate costs')
    cost_parser.add_argument('hours_per_day', type=float, help='Hours of usage per day')
    cost_parser.add_argument('--cost-per-hour', type=float, default=0.50, help='Cost per hour')

    args = parser.parse_args()

    # Check for command
    if not args.command:
        parser.print_help()
        return 0

    # Get API key
    api_key = args.api_key or os.getenv('RUNPOD_API_KEY')

    if not api_key and args.command != 'cost':
        print("❌ Error: No API key provided")
        print()
        print("Options:")
        print("  1. Pass --api-key YOUR_KEY")
        print("  2. Set RUNPOD_API_KEY environment variable")
        print()
        print("Get your API key from: https://runpod.io/settings")
        return 1

    # Initialize manager
    if args.command != 'cost':
        try:
            manager = RunPodManager(api_key=api_key)
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            return 1
    else:
        manager = RunPodManager(api_key="dummy")  # Cost doesn't need real key

    # Route to command handler
    commands = {
        'list': cmd_list,
        'create': cmd_create,
        'stop': cmd_stop,
        'resume': cmd_resume,
        'terminate': cmd_terminate,
        'status': cmd_status,
        'ssh': cmd_ssh,
        'tunnel': cmd_tunnel,
        'cost': cmd_cost
    }

    handler = commands.get(args.command)

    if handler:
        try:
            return handler(manager, args) or 0
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
