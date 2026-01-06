#!/usr/bin/env python3
"""
Real-time memory and swap monitor for RAG pipeline.

Usage:
    python monitor_memory.py

Press Ctrl+C to stop.
"""

import psutil
import time
import sys

def main():
    print("Monitoring system memory (Ctrl+C to stop)...")
    print("Watch for: RAM > 80% or Swap > 60%")
    print()

    try:
        while True:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            # Determine status
            if mem.percent < 75 and swap.percent < 50:
                status = "✅ HEALTHY"
            elif mem.percent < 85 and swap.percent < 60:
                status = "⚠️  MONITOR"
            else:
                status = "🔴 HIGH"

            # Print status line (overwrites previous)
            print(
                f"\r{status} | "
                f"RAM: {mem.percent:4.1f}% ({mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB) | "
                f"Swap: {swap.percent:4.1f}% ({swap.used/1024**3:.1f}/{swap.total/1024**3:.1f} GB)    ",
                end='',
                flush=True
            )

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print(f"Final: RAM {mem.percent:.1f}%, Swap {swap.percent:.1f}%")

if __name__ == "__main__":
    main()
