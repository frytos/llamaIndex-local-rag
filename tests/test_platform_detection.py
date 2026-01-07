"""
Unit tests for platform detection utility.

Tests:
- Platform detection
- GPU detection
- Mac chip detection
- Memory detection
- Git metadata extraction
"""

import os
import platform
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from utils.platform_detection import (
    detect_gpu_type,
    detect_mac_chip,
    detect_platform,
    get_git_metadata,
    get_platform_info,
    get_python_version,
    get_system_memory_gb,
)


class TestDetectPlatform:
    """Test platform detection."""

    def test_detect_platform_override(self):
        """Test that PERFORMANCE_PLATFORM override works."""
        with patch.dict(os.environ, {"PERFORMANCE_PLATFORM": "Custom_Platform"}):
            platform = detect_platform()
            assert platform == "Custom_Platform"

    def test_detect_github_actions(self):
        """Test GitHub Actions detection."""
        with patch.dict(
            os.environ, {"CI": "1", "GITHUB_ACTIONS": "true"}, clear=True
        ):
            with patch("platform.system", return_value="Darwin"):
                platform = detect_platform()
                assert platform == "GitHub_Actions_Darwin"

    def test_detect_runpod(self):
        """Test RunPod detection."""
        with patch.dict(
            os.environ, {"CI": "1", "RUNPOD_POD_ID": "abc123"}, clear=True
        ):
            with patch("utils.platform_detection.detect_gpu_type", return_value="A100"):
                platform = detect_platform()
                assert platform == "RunPod_A100"

    def test_detect_runpod_no_gpu(self):
        """Test RunPod detection without GPU."""
        with patch.dict(
            os.environ, {"CI": "1", "RUNPOD_POD_ID": "abc123"}, clear=True
        ):
            with patch("utils.platform_detection.detect_gpu_type", return_value=None):
                platform = detect_platform()
                assert platform == "RunPod_Unknown"

    def test_detect_generic_ci(self):
        """Test generic CI detection."""
        with patch.dict(os.environ, {"CI": "1"}, clear=True):
            platform = detect_platform()
            assert platform == "CI_Generic"

    @patch("platform.system", return_value="Darwin")
    @patch("platform.machine", return_value="arm64")
    def test_detect_m1_mac(self, mock_machine, mock_system):
        """Test M1 Mac detection."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "utils.platform_detection.get_system_memory_gb", return_value=16
            ):
                with patch(
                    "utils.platform_detection.detect_mac_chip", return_value="M1"
                ):
                    platform = detect_platform()
                    assert platform == "M1_Mac_16GB"

    @patch("platform.system", return_value="Linux")
    def test_detect_gpu_server(self, mock_system):
        """Test GPU server detection."""
        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "utils.platform_detection.detect_gpu_type", return_value="RTX_4090"
            ):
                with patch(
                    "utils.platform_detection.get_system_memory_gb", return_value=64
                ):
                    platform = detect_platform()
                    assert platform == "RTX_4090_64GB"

    @patch("platform.system", return_value="Linux")
    @patch("platform.machine", return_value="x86_64")
    def test_detect_fallback(self, mock_machine, mock_system):
        """Test fallback to generic platform."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("utils.platform_detection.detect_gpu_type", return_value=None):
                platform = detect_platform()
                assert platform == "Linux_x86_64"


class TestDetectGPUType:
    """Test GPU detection."""

    @patch("subprocess.run")
    def test_detect_gpu_nvidia_smi_rtx(self, mock_run):
        """Test NVIDIA GPU detection via nvidia-smi (RTX)."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="NVIDIA GeForce RTX 4090\n"
        )

        gpu = detect_gpu_type()
        assert gpu == "RTX_4090"

    @patch("subprocess.run")
    def test_detect_gpu_nvidia_smi_a100(self, mock_run):
        """Test NVIDIA GPU detection via nvidia-smi (A100)."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout="NVIDIA A100-SXM4-80GB\n"
        )

        gpu = detect_gpu_type()
        assert gpu == "A100"

    @patch("subprocess.run")
    def test_detect_gpu_nvidia_smi_h100(self, mock_run):
        """Test NVIDIA GPU detection via nvidia-smi (H100)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="NVIDIA H100 PCIe\n")

        gpu = detect_gpu_type()
        assert gpu == "H100"

    @patch("subprocess.run")
    def test_detect_gpu_nvidia_smi_failure(self, mock_run):
        """Test nvidia-smi failure handling."""
        mock_run.side_effect = FileNotFoundError()

        gpu = detect_gpu_type()
        # Should fall back to torch detection (mocked as None)
        assert gpu is None

    @patch("subprocess.run")
    def test_detect_gpu_torch_fallback(self, mock_run):
        """Test torch fallback when nvidia-smi fails."""
        mock_run.side_effect = FileNotFoundError()

        with patch("utils.platform_detection.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3090"

            gpu = detect_gpu_type()
            assert gpu == "RTX_3090"

    def test_detect_gpu_no_gpu(self):
        """Test when no GPU is available."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            # torch not available or no CUDA
            gpu = detect_gpu_type()
            # Should return None or handle gracefully


class TestDetectMacChip:
    """Test Mac chip detection."""

    @patch("subprocess.run")
    def test_detect_m1_chip(self, mock_run):
        """Test M1 chip detection."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M1 Pro\n")

        chip = detect_mac_chip()
        assert chip == "M1"

    @patch("subprocess.run")
    def test_detect_m2_chip(self, mock_run):
        """Test M2 chip detection."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M2 Max\n")

        chip = detect_mac_chip()
        assert chip == "M2"

    @patch("subprocess.run")
    def test_detect_m3_chip(self, mock_run):
        """Test M3 chip detection."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Apple M3\n")

        chip = detect_mac_chip()
        assert chip == "M3"

    @patch("subprocess.run")
    def test_detect_mac_chip_failure(self, mock_run):
        """Test fallback when sysctl fails."""
        mock_run.side_effect = FileNotFoundError()

        chip = detect_mac_chip()
        assert chip == "Mac"


class TestGetSystemMemory:
    """Test system memory detection."""

    def test_get_memory_psutil(self):
        """Test memory detection via psutil."""
        try:
            import psutil

            memory = get_system_memory_gb()
            assert isinstance(memory, int)
            assert memory > 0
            assert memory < 1024  # Sanity check (< 1TB)
        except ImportError:
            pytest.skip("psutil not available")

    @patch("subprocess.run")
    def test_get_memory_sysctl(self, mock_run):
        """Test memory detection via sysctl (macOS)."""
        # 16GB = 16 * 1024^3 bytes
        mock_run.return_value = MagicMock(returncode=0, stdout="17179869184\n")

        with patch("utils.platform_detection.psutil", None):
            memory = get_system_memory_gb()
            assert memory == 16

    def test_get_memory_proc_meminfo(self):
        """Test memory detection via /proc/meminfo (Linux)."""
        # Mock reading /proc/meminfo
        meminfo_content = "MemTotal:       16384000 kB\n"

        with patch("utils.platform_detection.psutil", None):
            with patch("subprocess.run", side_effect=FileNotFoundError()):
                with patch("builtins.open", create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value = meminfo_content.splitlines(
                        keepends=True
                    )
                    memory = get_system_memory_gb()
                    assert memory == 16

    def test_get_memory_fallback(self):
        """Test fallback to default value."""
        with patch("utils.platform_detection.psutil", None):
            with patch("subprocess.run", side_effect=FileNotFoundError()):
                with patch("builtins.open", side_effect=FileNotFoundError()):
                    memory = get_system_memory_gb()
                    assert memory == 16  # Default fallback


class TestGetGitMetadata:
    """Test git metadata extraction."""

    @patch("subprocess.run")
    def test_get_git_metadata_success(self, mock_run):
        """Test successful git metadata extraction."""
        # Mock git commands
        def run_side_effect(cmd, **kwargs):
            if "rev-parse" in cmd and "--short" in cmd:
                return MagicMock(returncode=0, stdout="abc123d\n")
            elif "rev-parse" in cmd and "--abbrev-ref" in cmd:
                return MagicMock(returncode=0, stdout="main\n")
            return MagicMock(returncode=1, stdout="")

        mock_run.side_effect = run_side_effect

        metadata = get_git_metadata()
        assert metadata["commit"] == "abc123d"
        assert metadata["branch"] == "main"

    @patch("subprocess.run")
    def test_get_git_metadata_no_git(self, mock_run):
        """Test when git is not available."""
        mock_run.side_effect = FileNotFoundError()

        metadata = get_git_metadata()
        assert metadata["commit"] is None
        assert metadata["branch"] is None

    @patch("subprocess.run")
    def test_get_git_metadata_not_repo(self, mock_run):
        """Test when not in a git repository."""
        mock_run.return_value = MagicMock(returncode=128, stdout="")

        metadata = get_git_metadata()
        assert metadata["commit"] is None
        assert metadata["branch"] is None


class TestGetPythonVersion:
    """Test Python version detection."""

    def test_get_python_version(self):
        """Test Python version string."""
        version = get_python_version()
        assert isinstance(version, str)
        # Should be like "3.11.9"
        assert "." in version


class TestGetPlatformInfo:
    """Test comprehensive platform info."""

    def test_get_platform_info_structure(self):
        """Test that platform info has all expected keys."""
        info = get_platform_info()

        required_keys = [
            "platform",
            "system",
            "machine",
            "python_version",
            "memory_gb",
            "gpu_type",
            "git_commit",
            "git_branch",
            "is_ci",
            "is_github_actions",
            "is_runpod",
        ]

        for key in required_keys:
            assert key in info

    def test_get_platform_info_types(self):
        """Test that platform info values have correct types."""
        info = get_platform_info()

        assert isinstance(info["platform"], str)
        assert isinstance(info["system"], str)
        assert isinstance(info["machine"], str)
        assert isinstance(info["python_version"], str)
        assert isinstance(info["memory_gb"], int)
        assert isinstance(info["is_ci"], bool)
        assert isinstance(info["is_github_actions"], bool)
        assert isinstance(info["is_runpod"], bool)

        # Optional fields
        assert info["gpu_type"] is None or isinstance(info["gpu_type"], str)
        assert info["git_commit"] is None or isinstance(info["git_commit"], str)
        assert info["git_branch"] is None or isinstance(info["git_branch"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
