"""
Auto-detect PostgreSQL connection details from RunPod API.

This allows the app to dynamically connect to RunPod pods without
hardcoding connection details in environment variables.

Usage:
    from utils.runpod_db_config import get_postgres_config

    config = get_postgres_config()
    if config:
        conn = psycopg2.connect(
            host=config['host'],
            port=config['port'],
            user=config['user'],
            password=config['password'],
            dbname=config['database']
        )
"""

import os
import logging
from typing import Optional, Dict

log = logging.getLogger(__name__)


def get_postgres_config(
    runpod_api_key: Optional[str] = None,
    pod_name_pattern: Optional[str] = None,
    pod_id: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Auto-detect PostgreSQL connection details from RunPod API.

    Priority order:
    1. If PGHOST is set and not "auto" → use static config
    2. If pod_id provided → use that specific pod
    3. If pod_name_pattern provided → find pod matching pattern
    4. Otherwise → use most recently created pod with PostgreSQL port

    Args:
        runpod_api_key: RunPod API key (or uses RUNPOD_API_KEY env var)
        pod_name_pattern: Pattern to match pod name (e.g., "rag-pipeline")
        pod_id: Specific pod ID to use

    Returns:
        Dict with connection config or None if unavailable

    Example:
        >>> config = get_postgres_config()
        >>> print(config)
        {
            'host': '38.65.239.5',
            'port': '18832',
            'user': 'fryt',
            'password': 'frytos',
            'database': 'vector_db',
            'sslmode': 'require'
        }
    """

    # Check if static config is provided (not auto-mode)
    pghost = os.getenv("PGHOST", "auto")

    if pghost and pghost != "auto":
        # Use static configuration from environment
        return {
            "host": pghost,
            "port": os.getenv("PGPORT", "5432"),
            "user": os.getenv("PGUSER", "postgres"),
            "password": os.getenv("PGPASSWORD", ""),
            "database": os.getenv("DB_NAME", "vector_db"),
            "sslmode": os.getenv("PGSSLMODE", "prefer"),
        }

    # Auto-mode: Fetch from RunPod API
    log.info("🔍 Auto-detecting PostgreSQL from RunPod API...")

    try:
        from utils.runpod_manager import RunPodManager

        api_key = runpod_api_key or os.getenv("RUNPOD_API_KEY")

        if not api_key:
            log.warning("RUNPOD_API_KEY not set - cannot auto-detect")
            return None

        manager = RunPodManager(api_key=api_key)

        # Get all pods
        pods = manager.list_pods()

        if not pods:
            log.warning("No RunPod pods found")
            return None

        # Filter and select pod
        target_pod = None

        if pod_id:
            # Use specific pod ID
            target_pod = manager.get_pod(pod_id)
            if target_pod:
                log.info(f"Using specified pod: {pod_id}")

        elif pod_name_pattern:
            # Find pod matching name pattern
            pattern = pod_name_pattern.lower()
            for pod in pods:
                name = pod.get("name", "").lower()
                if pattern in name:
                    target_pod = pod
                    log.info(f"Found pod matching pattern '{pattern}': {name}")
                    break

        else:
            # Use most recently created pod with PostgreSQL port
            # Sort by creation time (newest first)
            pods_sorted = sorted(
                pods,
                key=lambda p: p.get("createdAt", ""),
                reverse=True
            )

            for pod in pods_sorted:
                runtime = pod.get("runtime", {})
                ports = runtime.get("ports", [])

                # Check if pod has PostgreSQL port (5432)
                has_postgres = any(
                    port.get("privatePort") == 5432
                    for port in ports
                )

                if has_postgres:
                    target_pod = pod
                    log.info(f"Using most recent pod with PostgreSQL: {pod.get('name')}")
                    break

        if not target_pod:
            log.warning("No suitable RunPod pod found")
            return None

        # Extract PostgreSQL connection details
        runtime = target_pod.get("runtime", {})
        ports = runtime.get("ports", [])

        pg_host = None
        pg_port = None

        for port_info in ports:
            if port_info.get("privatePort") == 5432:
                pg_host = port_info.get("ip")
                pg_port = port_info.get("publicPort")
                break

        if not pg_host or not pg_port:
            log.warning("PostgreSQL port mapping not found in pod")
            return None

        # Get credentials from environment (or use defaults)
        config = {
            "host": pg_host,
            "port": str(pg_port),
            "user": os.getenv("PGUSER", "fryt"),
            "password": os.getenv("PGPASSWORD", "frytos"),
            "database": os.getenv("DB_NAME", "vector_db"),
            "sslmode": os.getenv("PGSSLMODE", "require"),
        }

        log.info(f"✅ Auto-detected PostgreSQL: {pg_host}:{pg_port}")

        return config

    except ImportError:
        log.error("runpod package not installed - cannot auto-detect")
        return None
    except Exception as e:
        log.error(f"Failed to auto-detect PostgreSQL: {e}")
        return None


def get_connection_string() -> Optional[str]:
    """
    Get PostgreSQL connection string with auto-detection.

    Returns:
        Connection string or None if unavailable

    Example:
        >>> conn_str = get_connection_string()
        >>> conn = psycopg2.connect(conn_str)
    """
    config = get_postgres_config()

    if not config:
        return None

    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
        f"?sslmode={config['sslmode']}"
    )
