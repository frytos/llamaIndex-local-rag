"""
RunPod Health Monitoring Utilities

Provides health checks for RunPod deployments:
- SSH connectivity
- Service status (PostgreSQL, vLLM)
- GPU metrics
- Network connectivity

Usage:
    from utils.runpod_health import check_vllm_health, check_postgres_health

    # Check vLLM server
    vllm_status = check_vllm_health(host="localhost", port=8000)
    print(f"vLLM: {vllm_status['status']}")

    # Check PostgreSQL
    pg_status = check_postgres_health(host="localhost", port=5432)
    print(f"PostgreSQL: {pg_status['status']}")
"""

import os
import socket
import subprocess
import time
from typing import Dict, List, Optional
import logging

log = logging.getLogger(__name__)


def check_ssh_connectivity(ssh_host: str, timeout: int = 10) -> bool:
    """
    Check if SSH is accessible.

    Args:
        ssh_host: SSH hostname (e.g., "abc123xyz")
        timeout: Connection timeout in seconds

    Returns:
        True if SSH is accessible, False otherwise

    Example:
        >>> check_ssh_connectivity("abc123xyz")
        True
    """
    try:
        # Test SSH connectivity
        cmd = [
            "ssh",
            "-o", "ConnectTimeout=10",
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            f"{ssh_host}@ssh.runpod.io",
            "echo 'connected'"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True
        )

        return result.returncode == 0

    except (subprocess.TimeoutExpired, Exception) as e:
        log.debug(f"SSH connectivity check failed: {e}")
        return False


def check_port_open(host: str, port: int, timeout: int = 5) -> bool:
    """
    Check if a port is open.

    Args:
        host: Hostname or IP
        port: Port number
        timeout: Connection timeout

    Returns:
        True if port is open, False otherwise

    Example:
        >>> check_port_open("localhost", 8000)
        True
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        log.debug(f"Port check failed for {host}:{port}: {e}")
        return False


def check_vllm_health(host: str = "localhost", port: int = 8000, timeout: int = 5) -> Dict:
    """
    Check vLLM server health.

    Args:
        host: Server hostname
        port: Server port
        timeout: Request timeout

    Returns:
        Dict with status, latency, and error info

    Example:
        >>> status = check_vllm_health()
        >>> print(f"vLLM: {status['status']}")
        >>> if status['status'] == 'healthy':
        ...     print(f"Latency: {status['latency_ms']}ms")
    """
    try:
        import requests

        start = time.time()
        response = requests.get(
            f"http://{host}:{port}/health",
            timeout=timeout
        )
        latency = (time.time() - start) * 1000

        if response.status_code == 200:
            return {
                "status": "healthy",
                "latency_ms": round(latency, 2),
                "response": response.text
            }
        else:
            return {
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}",
                "response": response.text
            }

    except requests.exceptions.ConnectionError:
        return {
            "status": "unreachable",
            "error": "Connection refused (server not running or port not forwarded)"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "error": f"Request timeout after {timeout}s"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_postgres_health(
    host: str = "localhost",
    port: int = 5432,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: str = "vector_db",
    timeout: int = 5
) -> Dict:
    """
    Check PostgreSQL health.

    Args:
        host: PostgreSQL hostname
        port: PostgreSQL port
        user: Database user
        password: Database password
        database: Database name
        timeout: Connection timeout

    Returns:
        Dict with status and error info

    Example:
        >>> status = check_postgres_health()
        >>> print(f"PostgreSQL: {status['status']}")
    """
    try:
        import psycopg2

        # Get credentials from env if not provided
        user = user or os.getenv("PGUSER", "fryt")
        password = password or os.getenv("PGPASSWORD", "frytos")

        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            connect_timeout=timeout
        )

        # Test basic query
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()

        # Check pgvector extension
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """)
            has_pgvector = cur.fetchone()[0]

        conn.close()

        return {
            "status": "healthy",
            "pgvector": has_pgvector,
            "version": "Connected successfully"
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_gpu_available(ssh_host: Optional[str] = None) -> Dict:
    """
    Check if GPU is available.

    Args:
        ssh_host: If provided, check via SSH

    Returns:
        Dict with GPU availability and metrics

    Example:
        >>> gpu_status = check_gpu_available(ssh_host="abc123")
        >>> if gpu_status['available']:
        ...     print(f"GPU: {gpu_status['name']}")
    """
    try:
        if ssh_host:
            # Check via SSH
            cmd = [
                "ssh",
                "-o", "ConnectTimeout=10",
                f"{ssh_host}@ssh.runpod.io",
                "nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=15,
                text=True
            )

            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    parts = output.split(',')
                    return {
                        "available": True,
                        "name": parts[0].strip(),
                        "memory_total": parts[1].strip(),
                        "memory_used": parts[2].strip(),
                        "via": "ssh"
                    }

        else:
            # Check locally
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                return {
                    "available": True,
                    "name": result.stdout.strip(),
                    "via": "local"
                }

        return {
            "available": False,
            "error": "nvidia-smi not available"
        }

    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


def wait_for_service(
    check_function,
    timeout: int = 300,
    check_interval: int = 10,
    service_name: str = "service",
    **kwargs
) -> bool:
    """
    Wait for a service to become healthy.

    Args:
        check_function: Health check function to call
        timeout: Maximum wait time in seconds
        check_interval: Seconds between checks
        service_name: Service name for logging
        **kwargs: Arguments to pass to check_function

    Returns:
        True if service is healthy, False on timeout

    Example:
        >>> wait_for_service(
        ...     check_vllm_health,
        ...     timeout=60,
        ...     service_name="vLLM",
        ...     host="localhost",
        ...     port=8000
        ... )
        True
    """
    start = time.time()
    elapsed = 0

    log.info(f"⏳ Waiting for {service_name} to be ready...")
    log.info(f"   Timeout: {timeout}s | Check interval: {check_interval}s")

    while elapsed < timeout:
        status = check_function(**kwargs)

        if status.get("status") == "healthy":
            log.info(f"✅ {service_name} is ready!")
            return True

        elapsed = time.time() - start
        remaining = timeout - elapsed

        log.info(
            f"   Status: {status.get('status', 'unknown')} | "
            f"Remaining: {remaining:.0f}s"
        )

        time.sleep(check_interval)
        elapsed = time.time() - start

    log.error(f"❌ {service_name} not ready after {timeout}s")
    return False


def wait_for_services(
    services: List[Dict],
    timeout: int = 300,
    check_interval: int = 10
) -> Dict[str, bool]:
    """
    Wait for multiple services to be ready.

    Args:
        services: List of service configs with 'name', 'check_function', and args
        timeout: Maximum wait time
        check_interval: Seconds between checks

    Returns:
        Dict mapping service names to ready status

    Example:
        >>> results = wait_for_services([
        ...     {
        ...         'name': 'vLLM',
        ...         'check_function': check_vllm_health,
        ...         'host': 'localhost',
        ...         'port': 8000
        ...     },
        ...     {
        ...         'name': 'PostgreSQL',
        ...         'check_function': check_postgres_health,
        ...         'host': 'localhost',
        ...         'port': 5432
        ...     }
        ... ])
        >>> print(f"vLLM ready: {results['vLLM']}")
    """
    results = {}

    for service_config in services:
        name = service_config.pop('name')
        check_function = service_config.pop('check_function')

        ready = wait_for_service(
            check_function,
            timeout=timeout,
            check_interval=check_interval,
            service_name=name,
            **service_config
        )

        results[name] = ready

    return results


def comprehensive_health_check(
    ssh_host: Optional[str] = None,
    local: bool = True
) -> Dict:
    """
    Run comprehensive health check.

    Args:
        ssh_host: SSH hostname for remote checks
        local: Check local services (assumes SSH tunnel or local deployment)

    Returns:
        Dict with all health check results

    Example:
        >>> health = comprehensive_health_check(local=True)
        >>> print(f"Overall: {health['overall_status']}")
        >>> for service, status in health['services'].items():
        ...     print(f"{service}: {status['status']}")
    """
    results = {
        "services": {},
        "overall_status": "unknown",
        "timestamp": time.time()
    }

    if local:
        # Check local services (via SSH tunnel)
        log.info("Running local health checks (requires SSH tunnel)...")

        # vLLM
        log.info("Checking vLLM server...")
        results["services"]["vllm"] = check_vllm_health()

        # PostgreSQL
        log.info("Checking PostgreSQL...")
        results["services"]["postgres"] = check_postgres_health()

    if ssh_host:
        # Check SSH connectivity
        log.info("Checking SSH connectivity...")
        ssh_ok = check_ssh_connectivity(ssh_host)
        results["services"]["ssh"] = {
            "status": "healthy" if ssh_ok else "unhealthy"
        }

        # Check GPU
        log.info("Checking GPU availability...")
        results["services"]["gpu"] = check_gpu_available(ssh_host=ssh_host)

    # Determine overall status
    statuses = [
        s.get("status") == "healthy"
        for s in results["services"].values()
        if "status" in s
    ]

    if all(statuses):
        results["overall_status"] = "healthy"
    elif any(statuses):
        results["overall_status"] = "partial"
    else:
        results["overall_status"] = "unhealthy"

    return results


def print_health_report(health: Dict):
    """
    Print formatted health check report.

    Args:
        health: Health check results from comprehensive_health_check()

    Example:
        >>> health = comprehensive_health_check()
        >>> print_health_report(health)
    """
    print("\n" + "=" * 70)
    print("HEALTH CHECK REPORT")
    print("=" * 70)
    print()

    for service_name, status in health["services"].items():
        service_status = status.get("status", "unknown")

        if service_status == "healthy":
            icon = "✅"
        elif service_status == "unhealthy":
            icon = "❌"
        else:
            icon = "⚠️ "

        print(f"{icon} {service_name.upper()}: {service_status}")

        # Print additional details
        if "latency_ms" in status:
            print(f"   Latency: {status['latency_ms']}ms")

        if "error" in status:
            print(f"   Error: {status['error']}")

        if "pgvector" in status:
            pgvector_status = "✅" if status["pgvector"] else "❌"
            print(f"   pgvector extension: {pgvector_status}")

        if "name" in status and service_name == "gpu":
            print(f"   GPU: {status['name']}")
            if "memory_total" in status:
                print(f"   Memory: {status['memory_used']} / {status['memory_total']}")

        print()

    # Overall status
    print("=" * 70)
    overall = health["overall_status"]

    if overall == "healthy":
        print("✅ Overall Status: ALL SYSTEMS OPERATIONAL")
    elif overall == "partial":
        print("⚠️  Overall Status: SOME SERVICES DOWN")
    else:
        print("❌ Overall Status: SYSTEM UNHEALTHY")

    print("=" * 70)
    print()
