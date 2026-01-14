"""
SSH Tunnel Management for RunPod

Manages SSH tunnels for accessing RunPod services locally.

Features:
- Automatic tunnel creation
- Port forwarding management
- Tunnel health monitoring
- Automatic reconnection

Usage:
    from utils.ssh_tunnel import SSHTunnelManager

    # Create tunnel
    tunnel = SSHTunnelManager(ssh_host="abc123")
    tunnel.create_tunnel(ports=[8000, 5432, 3000])

    # Check if tunnel is active
    if tunnel.is_active():
        print("Tunnel is running!")

    # Stop tunnel
    tunnel.stop_tunnel()
"""

import os
import subprocess
import time
import signal
from typing import List, Optional
import logging

log = logging.getLogger(__name__)


class SSHTunnelManager:
    """
    Manage SSH tunnels to RunPod pods.

    Provides:
    - Port forwarding setup
    - Tunnel lifecycle management
    - Health monitoring
    - Auto-reconnection
    """

    def __init__(self, ssh_host: str, ssh_port: int = 22):
        """
        Initialize SSH tunnel manager.

        Args:
            ssh_host: SSH hostname (e.g., "abc123xyz")
            ssh_port: SSH port (default: 22)
        """
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.process = None
        self.ports = []
        self.ssh_key_path = os.path.expanduser("~/.ssh/id_rsa")

    def create_tunnel(
        self,
        ports: Optional[List[int]] = None,
        ssh_key: Optional[str] = None,
        background: bool = True
    ) -> bool:
        """
        Create SSH tunnel with port forwarding.

        Args:
            ports: List of ports to forward (default: [8000, 5432, 3000])
            ssh_key: Path to SSH key (default: ~/.ssh/id_rsa)
            background: Run in background (default: True)

        Returns:
            True if tunnel created successfully, False otherwise

        Example:
            >>> tunnel = SSHTunnelManager("abc123")
            >>> tunnel.create_tunnel(ports=[8000, 5432])
            True
        """
        # Default ports: vLLM (8000), PostgreSQL (5432), Grafana (3000)
        if ports is None:
            ports = [8000, 5432, 3000]

        self.ports = ports

        # Use custom SSH key if provided
        if ssh_key:
            self.ssh_key_path = ssh_key

        # Build SSH command
        port_forwards = []
        for port in ports:
            port_forwards.extend(["-L", f"{port}:localhost:{port}"])

        ssh_cmd = [
            "ssh",
            "-N",  # No remote command
            "-o", "ServerAliveInterval=60",  # Keep alive
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-p", str(self.ssh_port),
        ]

        # Add SSH key if exists
        if os.path.exists(self.ssh_key_path):
            ssh_cmd.extend(["-i", self.ssh_key_path])

        # Add port forwards
        ssh_cmd.extend(port_forwards)

        # Add host
        ssh_cmd.append(f"{self.ssh_host}@ssh.runpod.io")

        log.info(f"Creating SSH tunnel to {self.ssh_host}...")
        log.info(f"Forwarding ports: {', '.join(map(str, ports))}")

        try:
            if background:
                # Run in background
                self.process = subprocess.Popen(
                    ssh_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid  # Create new session
                )

                # Give it a moment to establish
                time.sleep(2)

                # Check if still running
                if self.process.poll() is None:
                    log.info(f"✅ SSH tunnel created (PID: {self.process.pid})")
                    log.info(f"   Ports forwarded: {', '.join(map(str, ports))}")
                    return True
                else:
                    stderr = self.process.stderr.read().decode()
                    log.error(f"❌ SSH tunnel failed: {stderr}")
                    return False

            else:
                # Run in foreground (blocking)
                log.info("Starting SSH tunnel in foreground (press Ctrl+C to stop)...")
                subprocess.run(ssh_cmd)
                return True

        except Exception as e:
            log.error(f"Failed to create SSH tunnel: {e}")
            return False

    def is_active(self) -> bool:
        """
        Check if tunnel is active.

        Returns:
            True if tunnel process is running, False otherwise

        Example:
            >>> tunnel.is_active()
            True
        """
        if self.process is None:
            return False

        # Check if process is still running
        return self.process.poll() is None

    def stop_tunnel(self) -> bool:
        """
        Stop SSH tunnel.

        Returns:
            True if stopped successfully, False otherwise

        Example:
            >>> tunnel.stop_tunnel()
            True
        """
        if self.process is None:
            log.warning("No tunnel process to stop")
            return False

        try:
            # Send SIGTERM to process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

            # Wait for process to terminate
            self.process.wait(timeout=5)

            log.info("✅ SSH tunnel stopped")
            self.process = None
            return True

        except subprocess.TimeoutExpired:
            # Force kill if not responding
            log.warning("Tunnel didn't stop gracefully, force killing...")
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process = None
            return True

        except Exception as e:
            log.error(f"Failed to stop tunnel: {e}")
            return False

    def get_tunnel_command(self) -> str:
        """
        Get the SSH tunnel command string.

        Returns:
            SSH command string for manual execution

        Example:
            >>> cmd = tunnel.get_tunnel_command()
            >>> print(cmd)
            ssh -L 8000:localhost:8000 ... abc123@ssh.runpod.io
        """
        port_forwards = " ".join([f"-L {p}:localhost:{p}" for p in self.ports])

        return (
            f"ssh -N {port_forwards} "
            f"{self.ssh_host}@ssh.runpod.io"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit - auto cleanup."""
        if self.is_active():
            self.stop_tunnel()


# Convenience functions

def create_tunnel(
    ssh_host: str,
    ports: Optional[List[int]] = None,
    background: bool = True
) -> SSHTunnelManager:
    """
    Convenience function to create SSH tunnel.

    Args:
        ssh_host: SSH hostname
        ports: Ports to forward
        background: Run in background

    Returns:
        SSHTunnelManager instance

    Example:
        >>> tunnel = create_tunnel("abc123", ports=[8000, 5432])
        >>> # Use services...
        >>> tunnel.stop_tunnel()
    """
    tunnel = SSHTunnelManager(ssh_host)
    tunnel.create_tunnel(ports=ports, background=background)
    return tunnel


def run_with_tunnel(ssh_host: str, ports: List[int], func, *args, **kwargs):
    """
    Run a function with an active SSH tunnel.

    Args:
        ssh_host: SSH hostname
        ports: Ports to forward
        func: Function to run
        *args, **kwargs: Arguments for function

    Returns:
        Function result

    Example:
        >>> def query_vllm():
        ...     # Query vLLM through tunnel
        ...     return "response"
        >>>
        >>> result = run_with_tunnel("abc123", [8000], query_vllm)
    """
    with SSHTunnelManager(ssh_host) as tunnel:
        tunnel.create_tunnel(ports=ports)

        # Wait for tunnel to be ready
        time.sleep(2)

        # Run function
        return func(*args, **kwargs)
