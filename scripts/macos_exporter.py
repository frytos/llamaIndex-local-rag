#!/usr/bin/env python3
"""
macOS System Metrics Exporter for Prometheus

Exposes real macOS system metrics (CPU, memory, disk, swap) in Prometheus format.
This works around the limitation that node_exporter in Docker only sees the Docker VM.

Usage:
    python macos_exporter.py [--port 9101]

Metrics exposed:
    - macos_cpu_percent: CPU usage percentage
    - macos_cpu_count: Number of CPU cores
    - macos_load_avg_1m: Load average (1 minute)
    - macos_load_avg_5m: Load average (5 minutes)
    - macos_load_avg_15m: Load average (15 minutes)
    - macos_memory_total_bytes: Total physical memory
    - macos_memory_available_bytes: Available memory
    - macos_memory_used_bytes: Used memory
    - macos_memory_percent: Memory usage percentage
    - macos_swap_total_bytes: Total swap space
    - macos_swap_used_bytes: Used swap space
    - macos_swap_percent: Swap usage percentage
    - macos_disk_usage_percent{path="/"}: Disk usage percentage by path
    - macos_process_count: Total number of running processes
"""

import argparse
import logging
import os
import platform
import time
from typing import Dict, Any

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed")
    print("Install with: pip install psutil")
    exit(1)

try:
    from prometheus_client import start_http_server, Gauge, Info
    from prometheus_client import REGISTRY
except ImportError:
    print("ERROR: prometheus_client not installed")
    print("Install with: pip install prometheus-client")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# Define Prometheus metrics
system_info = Info('macos_system', 'System information')

# CPU metrics
cpu_percent = Gauge('macos_cpu_percent', 'CPU usage percentage')
cpu_count_physical = Gauge('macos_cpu_count_physical', 'Number of physical CPU cores')
cpu_count_logical = Gauge('macos_cpu_count_logical', 'Number of logical CPU cores')
cpu_freq = Gauge('macos_cpu_freq_mhz', 'Current CPU frequency in MHz')
load_avg_1m = Gauge('macos_load_avg_1m', 'Load average (1 minute)')
load_avg_5m = Gauge('macos_load_avg_5m', 'Load average (5 minutes)')
load_avg_15m = Gauge('macos_load_avg_15m', 'Load average (15 minutes)')

# Memory metrics
memory_total = Gauge('macos_memory_total_bytes', 'Total physical memory in bytes')
memory_available = Gauge('macos_memory_available_bytes', 'Available memory in bytes')
memory_used = Gauge('macos_memory_used_bytes', 'Used memory in bytes')
memory_free = Gauge('macos_memory_free_bytes', 'Free memory in bytes')
memory_percent = Gauge('macos_memory_percent', 'Memory usage percentage')
memory_active = Gauge('macos_memory_active_bytes', 'Active memory in bytes')
memory_inactive = Gauge('macos_memory_inactive_bytes', 'Inactive memory in bytes')
memory_wired = Gauge('macos_memory_wired_bytes', 'Wired memory in bytes')

# Swap metrics
swap_total = Gauge('macos_swap_total_bytes', 'Total swap space in bytes')
swap_used = Gauge('macos_swap_used_bytes', 'Used swap space in bytes')
swap_free = Gauge('macos_swap_free_bytes', 'Free swap space in bytes')
swap_percent = Gauge('macos_swap_percent', 'Swap usage percentage')
swap_sin = Gauge('macos_swap_sin_bytes', 'Bytes swapped in')
swap_sout = Gauge('macos_swap_sout_bytes', 'Bytes swapped out')

# Disk metrics
disk_usage_percent = Gauge('macos_disk_usage_percent', 'Disk usage percentage', ['path', 'device'])
disk_total = Gauge('macos_disk_total_bytes', 'Total disk space in bytes', ['path', 'device'])
disk_used = Gauge('macos_disk_used_bytes', 'Used disk space in bytes', ['path', 'device'])
disk_free = Gauge('macos_disk_free_bytes', 'Free disk space in bytes', ['path', 'device'])

# Process metrics
process_count = Gauge('macos_process_count', 'Total number of running processes')
process_running = Gauge('macos_process_running_count', 'Number of running processes')
process_sleeping = Gauge('macos_process_sleeping_count', 'Number of sleeping processes')

# Network metrics (optional)
network_bytes_sent = Gauge('macos_network_bytes_sent_total', 'Total bytes sent', ['interface'])
network_bytes_recv = Gauge('macos_network_bytes_recv_total', 'Total bytes received', ['interface'])

# Exporter health
exporter_up = Gauge('macos_exporter_up', 'Exporter is running (always 1)')
exporter_scrape_duration = Gauge('macos_exporter_scrape_duration_seconds', 'Time taken to collect metrics')


def collect_system_info():
    """Collect static system information"""
    try:
        info_dict = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        system_info.info(info_dict)
        log.info(f"System info: {info_dict}")
    except Exception as e:
        log.error(f"Failed to collect system info: {e}")


def collect_cpu_metrics():
    """Collect CPU metrics"""
    try:
        # CPU percentage
        cpu_percent.set(psutil.cpu_percent(interval=0.1))

        # CPU count
        cpu_count_physical.set(psutil.cpu_count(logical=False) or 0)
        cpu_count_logical.set(psutil.cpu_count(logical=True) or 0)

        # CPU frequency
        freq = psutil.cpu_freq()
        if freq:
            cpu_freq.set(freq.current)

        # Load average
        load_1, load_5, load_15 = os.getloadavg()
        load_avg_1m.set(load_1)
        load_avg_5m.set(load_5)
        load_avg_15m.set(load_15)
    except Exception as e:
        log.error(f"Failed to collect CPU metrics: {e}")


def collect_memory_metrics():
    """Collect memory metrics"""
    try:
        mem = psutil.virtual_memory()
        memory_total.set(mem.total)
        memory_available.set(mem.available)
        memory_used.set(mem.used)
        memory_free.set(mem.free)
        memory_percent.set(mem.percent)

        # macOS-specific
        if hasattr(mem, 'active'):
            memory_active.set(mem.active)
        if hasattr(mem, 'inactive'):
            memory_inactive.set(mem.inactive)
        if hasattr(mem, 'wired'):
            memory_wired.set(mem.wired)
    except Exception as e:
        log.error(f"Failed to collect memory metrics: {e}")


def collect_swap_metrics():
    """Collect swap metrics"""
    try:
        swap = psutil.swap_memory()
        swap_total.set(swap.total)
        swap_used.set(swap.used)
        swap_free.set(swap.free)
        swap_percent.set(swap.percent)
        swap_sin.set(swap.sin)
        swap_sout.set(swap.sout)
    except Exception as e:
        log.error(f"Failed to collect swap metrics: {e}")


def collect_disk_metrics():
    """Collect disk metrics"""
    try:
        # Get all disk partitions
        partitions = psutil.disk_partitions()

        for partition in partitions:
            # Skip special mounts
            if partition.mountpoint.startswith(('/dev', '/sys', '/proc')):
                continue

            try:
                usage = psutil.disk_usage(partition.mountpoint)

                # Extract device name (remove /dev/ prefix if present)
                device = partition.device.replace('/dev/', '')

                disk_usage_percent.labels(path=partition.mountpoint, device=device).set(usage.percent)
                disk_total.labels(path=partition.mountpoint, device=device).set(usage.total)
                disk_used.labels(path=partition.mountpoint, device=device).set(usage.used)
                disk_free.labels(path=partition.mountpoint, device=device).set(usage.free)
            except PermissionError:
                # Skip partitions we can't access
                continue
    except Exception as e:
        log.error(f"Failed to collect disk metrics: {e}")


def collect_process_metrics():
    """Collect process metrics"""
    try:
        processes = list(psutil.process_iter(['status']))
        process_count.set(len(processes))

        # Count by status
        status_counts = {}
        for proc in processes:
            try:
                status = proc.info['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        process_running.set(status_counts.get(psutil.STATUS_RUNNING, 0))
        process_sleeping.set(status_counts.get(psutil.STATUS_SLEEPING, 0))
    except Exception as e:
        log.error(f"Failed to collect process metrics: {e}")


def collect_network_metrics():
    """Collect network metrics"""
    try:
        net_io = psutil.net_io_counters(pernic=True)

        for interface, stats in net_io.items():
            # Skip loopback
            if interface == 'lo0':
                continue

            network_bytes_sent.labels(interface=interface).set(stats.bytes_sent)
            network_bytes_recv.labels(interface=interface).set(stats.bytes_recv)
    except Exception as e:
        log.error(f"Failed to collect network metrics: {e}")


def collect_all_metrics():
    """Collect all metrics"""
    start_time = time.time()

    try:
        collect_cpu_metrics()
        collect_memory_metrics()
        collect_swap_metrics()
        collect_disk_metrics()
        collect_process_metrics()
        collect_network_metrics()

        exporter_up.set(1)
    except Exception as e:
        log.error(f"Failed to collect metrics: {e}")
    finally:
        duration = time.time() - start_time
        exporter_scrape_duration.set(duration)


def metrics_collection_loop(interval: int = 10):
    """Continuously collect metrics"""
    log.info(f"Starting metrics collection loop (interval: {interval}s)")

    while True:
        try:
            collect_all_metrics()
            time.sleep(interval)
        except KeyboardInterrupt:
            log.info("Shutting down metrics collection")
            break
        except Exception as e:
            log.error(f"Error in collection loop: {e}")
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description='macOS System Metrics Exporter for Prometheus'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9101,
        help='Port to expose metrics (default: 9101)'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Metrics collection interval in seconds (default: 10)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Collect static info once
    collect_system_info()

    # Start HTTP server
    log.info(f"Starting metrics server on port {args.port}")
    start_http_server(args.port)
    log.info(f"Metrics available at http://localhost:{args.port}/metrics")

    # Start collection loop
    metrics_collection_loop(interval=args.interval)


if __name__ == '__main__':
    main()
