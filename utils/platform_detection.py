"""
Platform Detection Utility

Auto-detect hardware platform for performance baseline comparison.
Supports M1 Mac, NVIDIA GPUs, GitHub Actions, RunPod, and generic platforms.

Usage:
    from utils.platform_detection import detect_platform, get_git_metadata

    platform = detect_platform()
    # Returns: "M1_Mac_16GB", "RTX_4090_24GB", "GitHub_Actions_macOS", etc.

    git_info = get_git_metadata()
    # Returns: {"commit": "abc123", "branch": "main"}

Environment Variables:
    PERFORMANCE_PLATFORM=M1_Mac_16GB  # Override auto-detection
    CI=1                               # CI environment indicator
    GITHUB_ACTIONS=true                # GitHub Actions CI
    RUNPOD_POD_ID=xyz                  # RunPod environment
"""

import logging
import os
import platform as platform_module
import subprocess
from typing import Dict, Optional

log = logging.getLogger(__name__)


def detect_platform() -> str:
    """
    Auto-detect hardware platform.

    Returns platform identifier like:
    - "M1_Mac_16GB"
    - "RTX_4090_24GB"
    - "GitHub_Actions_macOS"
    - "RunPod_A100_80GB"
    - "Darwin_arm64" (fallback)

    Environment:
        PERFORMANCE_PLATFORM: Override detection
    """
    # Check for override
    override = os.getenv("PERFORMANCE_PLATFORM")
    if override:
        log.info(f"Using platform override: {override}")
        return override

    # Check CI environments
    if os.getenv("CI"):
        if os.getenv("GITHUB_ACTIONS"):
            # GitHub Actions
            os_name = platform_module.system()
            return f"GitHub_Actions_{os_name}"

        elif os.getenv("RUNPOD_POD_ID"):
            # RunPod
            gpu_type = detect_gpu_type()
            if gpu_type:
                return f"RunPod_{gpu_type}"
            return "RunPod_Unknown"

        else:
            # Generic CI
            return "CI_Generic"

    # Check for Apple Silicon Mac
    if (
        platform_module.system() == "Darwin"
        and platform_module.machine() == "arm64"
    ):
        memory_gb = get_system_memory_gb()
        # Detect chip (M1, M2, M3, etc.)
        chip = detect_mac_chip()
        return f"{chip}_Mac_{memory_gb}GB"

    # Check for NVIDIA GPU
    gpu_type = detect_gpu_type()
    if gpu_type:
        memory_gb = get_system_memory_gb()
        return f"{gpu_type}_{memory_gb}GB"

    # Fallback to generic
    system = platform_module.system()
    machine = platform_module.machine()
    return f"{system}_{machine}"


def detect_gpu_type() -> Optional[str]:
    """
    Detect GPU type using nvidia-smi or torch.

    Returns:
        GPU type string (e.g., "RTX_4090", "A100", "H100") or None
    """
    # Try nvidia-smi first
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode == 0:
            gpu_name = result.stdout.strip().split("\n")[0]
            # Simplify name
            # "NVIDIA GeForce RTX 4090" -> "RTX_4090"
            # "NVIDIA A100-SXM4-80GB" -> "A100"
            gpu_name = gpu_name.replace("NVIDIA", "").replace("GeForce", "")
            gpu_name = gpu_name.strip().replace(" ", "_").replace("-", "_")

            # Extract main identifier
            if "RTX" in gpu_name:
                # RTX_4090, RTX_3090, etc.
                parts = [p for p in gpu_name.split("_") if p]
                if len(parts) >= 2 and parts[0] == "RTX":
                    return f"RTX_{parts[1]}"
            elif "A100" in gpu_name:
                return "A100"
            elif "H100" in gpu_name:
                return "H100"
            elif "V100" in gpu_name:
                return "V100"
            elif "T4" in gpu_name:
                return "T4"

            # Return simplified name
            return gpu_name[:20]  # Truncate long names

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Try torch
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Similar simplification
            if "RTX" in gpu_name:
                parts = gpu_name.split()
                for i, part in enumerate(parts):
                    if part == "RTX" and i + 1 < len(parts):
                        return f"RTX_{parts[i+1]}"
            elif "A100" in gpu_name:
                return "A100"
            elif "H100" in gpu_name:
                return "H100"

            return "GPU_Unknown"
    except ImportError:
        pass

    return None


def detect_mac_chip() -> str:
    """
    Detect Mac chip type (M1, M2, M3, etc.).

    Returns:
        Chip identifier (e.g., "M1", "M2", "M3") or "Mac"
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=1,
        )

        if result.returncode == 0:
            brand = result.stdout.strip()
            # "Apple M1 Pro" -> "M1"
            # "Apple M2 Max" -> "M2"
            # "Apple M3" -> "M3"
            if "M1" in brand:
                return "M1"
            elif "M2" in brand:
                return "M2"
            elif "M3" in brand:
                return "M3"
            elif "M4" in brand:
                return "M4"

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return "Mac"


def get_system_memory_gb() -> int:
    """
    Get total system memory in GB.

    Returns:
        Memory in GB (rounded to nearest int)
    """
    try:
        import psutil

        memory_bytes = psutil.virtual_memory().total
        memory_gb = int(round(memory_bytes / (1024**3)))
        return memory_gb

    except ImportError:
        # psutil not available, try platform-specific methods
        pass

    # Try sysctl (macOS, BSD)
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=1,
        )

        if result.returncode == 0:
            memory_bytes = int(result.stdout.strip())
            memory_gb = int(round(memory_bytes / (1024**3)))
            return memory_gb

    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Try /proc/meminfo (Linux)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # MemTotal:       16384000 kB
                    parts = line.split()
                    memory_kb = int(parts[1])
                    memory_gb = int(round(memory_kb / (1024**2)))
                    return memory_gb

    except (FileNotFoundError, ValueError):
        pass

    # Fallback
    log.warning("Unable to detect system memory, defaulting to 16GB")
    return 16


def get_git_metadata() -> Dict[str, str]:
    """
    Get current git commit and branch.

    Returns:
        Dictionary with "commit" and "branch" keys

    Example:
        {"commit": "abc123d", "branch": "main"}
    """
    metadata = {"commit": None, "branch": None}

    # Get commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            metadata["commit"] = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError):
        log.debug("Unable to get git commit")

    # Get branch name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.getcwd(),
        )

        if result.returncode == 0:
            metadata["branch"] = result.stdout.strip()

    except (subprocess.TimeoutExpired, FileNotFoundError):
        log.debug("Unable to get git branch")

    return metadata


def get_python_version() -> str:
    """
    Get Python version string.

    Returns:
        Version string (e.g., "3.11.9")
    """
    return platform_module.python_version()


def get_platform_info() -> Dict:
    """
    Get comprehensive platform information.

    Returns:
        Dictionary with all platform details
    """
    platform = detect_platform()
    git = get_git_metadata()

    return {
        "platform": platform,
        "system": platform_module.system(),
        "machine": platform_module.machine(),
        "python_version": get_python_version(),
        "memory_gb": get_system_memory_gb(),
        "gpu_type": detect_gpu_type(),
        "git_commit": git["commit"],
        "git_branch": git["branch"],
        "is_ci": bool(os.getenv("CI")),
        "is_github_actions": bool(os.getenv("GITHUB_ACTIONS")),
        "is_runpod": bool(os.getenv("RUNPOD_POD_ID")),
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Platform Detection")
    print("=" * 70)

    # Detect platform
    platform = detect_platform()
    print(f"Platform: {platform}")

    # Get detailed info
    info = get_platform_info()
    print(f"\nDetailed Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Git metadata
    git = get_git_metadata()
    print(f"\nGit Metadata:")
    print(f"  Commit: {git['commit']}")
    print(f"  Branch: {git['branch']}")

    # System info
    print(f"\nSystem Info:")
    print(f"  OS: {platform_module.system()}")
    print(f"  Machine: {platform_module.machine()}")
    print(f"  Python: {get_python_version()}")
    print(f"  Memory: {get_system_memory_gb()} GB")

    # GPU info
    gpu = detect_gpu_type()
    if gpu:
        print(f"  GPU: {gpu}")
    else:
        print(f"  GPU: None detected")

    # Environment
    print(f"\nEnvironment:")
    print(f"  CI: {bool(os.getenv('CI'))}")
    print(f"  GitHub Actions: {bool(os.getenv('GITHUB_ACTIONS'))}")
    print(f"  RunPod: {bool(os.getenv('RUNPOD_POD_ID'))}")
