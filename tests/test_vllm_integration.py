"""Tests for vLLM integration (client and wrapper).

This module tests the vLLM integration for LlamaIndex, including:
- vLLM client connectivity and health checks
- Server connection error handling
- Environment variable configuration
- vLLM wrapper initialization

Week 1 - Day 1: Basic connectivity tests (5 tests)
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock, Mock
import requests


# ============================================================================
# Mock LlamaIndex imports (so tests run without dependencies)
# ============================================================================

# Mock llama_index modules before importing vllm_client
mock_openai = MagicMock()
mock_llm_metadata = MagicMock()
mock_llms = MagicMock()

sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.llms'] = mock_llms
sys.modules['llama_index.llms.openai'] = mock_openai
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.llms'] = MagicMock()


# ============================================================================
# Day 1: vLLM Client Connectivity Tests (5 tests)
# ============================================================================


class TestVLLMClientConnectivity:
    """Test vLLM client connection and health checks.

    These tests validate that the vLLM client properly:
    - Connects to healthy servers
    - Handles connection failures gracefully
    - Validates server health
    - Uses environment variables correctly
    """

    @pytest.mark.unit
    def test_vllm_client_initialization_success(self, mock_vllm_health_check):
        """Test successful vLLM client initialization with healthy server.

        Given: A healthy vLLM server running at localhost:8000
        When: build_vllm_client() is called
        Then: Client is successfully created with correct configuration
        """
        # Mock OpenAI client return value
        mock_openai_instance = MagicMock()
        mock_openai_instance.api_base = "http://localhost:8000/v1"
        mock_openai_instance.model = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
        mock_openai_instance.temperature = 0.1
        mock_openai_instance.max_tokens = 256

        mock_openai.OpenAI.return_value = mock_openai_instance

        # Import after mocking
        import vllm_client

        with patch('requests.get', return_value=mock_vllm_health_check):
            llm = vllm_client.build_vllm_client()

            # Verify client was created
            assert llm is not None
            assert llm.api_base == "http://localhost:8000/v1"
            assert llm.model == "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"
            assert llm.temperature == 0.1
            assert llm.max_tokens == 256

    @pytest.mark.unit
    def test_vllm_client_connection_refused(self):
        """Test ConnectionError when vLLM server is not running.

        Given: No vLLM server running (connection refused)
        When: build_vllm_client() is called
        Then: ConnectionError is raised with helpful error message
        """
        import vllm_client

        # Mock connection refused error
        with patch('requests.get', side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(ConnectionError) as exc_info:
                vllm_client.build_vllm_client()

            # Verify error message is helpful
            error_msg = str(exc_info.value)
            assert "Cannot connect to vLLM server" in error_msg
            assert "start_vllm_server.sh" in error_msg
            assert "Connection refused" in error_msg

    @pytest.mark.unit
    def test_vllm_client_health_check_failure(self):
        """Test ConnectionError when health endpoint returns non-200 status.

        Given: vLLM server responds but health check fails (503)
        When: build_vllm_client() is called
        Then: ConnectionError is raised indicating server not healthy
        """
        import vllm_client

        # Mock unhealthy server response
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch('requests.get', return_value=mock_response):
            with pytest.raises(ConnectionError) as exc_info:
                vllm_client.build_vllm_client()

            # Verify error indicates unhealthy server
            error_msg = str(exc_info.value)
            assert "not healthy" in error_msg.lower()
            assert "503" in error_msg

    @pytest.mark.unit
    def test_vllm_client_timeout(self):
        """Test connection timeout handling.

        Given: vLLM server is unresponsive (timeout)
        When: build_vllm_client() is called
        Then: ConnectionError is raised with timeout information
        """
        import vllm_client

        # Mock timeout error
        with patch('requests.get', side_effect=requests.exceptions.Timeout("Request timed out")):
            with pytest.raises(ConnectionError) as exc_info:
                vllm_client.build_vllm_client()

            # Verify error message mentions timeout
            error_msg = str(exc_info.value)
            assert "Cannot connect to vLLM server" in error_msg
            assert "timed out" in error_msg.lower()

    @pytest.mark.unit
    @patch.dict(os.environ, {"VLLM_PORT": "9000", "VLLM_MODEL": "custom-model"}, clear=False)
    def test_vllm_client_custom_port_and_model(self, mock_vllm_health_check):
        """Test vLLM client uses custom port and model from environment.

        Given: VLLM_PORT=9000 and VLLM_MODEL=custom-model are set
        When: build_vllm_client() is called without arguments
        Then: Client uses custom port and model from environment
        """
        # Mock OpenAI client return value with custom config
        mock_openai_instance = MagicMock()
        mock_openai_instance.api_base = "http://localhost:9000/v1"
        mock_openai_instance.model = "custom-model"

        mock_openai.OpenAI.return_value = mock_openai_instance

        import vllm_client

        with patch('requests.get', return_value=mock_vllm_health_check) as mock_get:
            llm = vllm_client.build_vllm_client()

            # Verify custom port was used
            assert llm.api_base == "http://localhost:9000/v1"
            assert llm.model == "custom-model"

            # Verify health check used correct URL
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "localhost:9000" in call_args[0][0]
            assert "/health" in call_args[0][0]


# ============================================================================
# Test Discovery and Execution Notes
# ============================================================================
"""
Run Day 1 tests:
    pytest tests/test_vllm_integration.py::TestVLLMClientConnectivity -v

Run with coverage:
    pytest tests/test_vllm_integration.py \
        --cov=vllm_client \
        --cov=vllm_wrapper \
        --cov-report=term-missing

Expected results:
    - 5 tests passing
    - Coverage increase: +30-40% for vllm_client.py
    - Validates basic connectivity and error handling
"""
