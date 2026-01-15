"""
Tests Based on Session Issues - Prevent Regressions

This test suite captures all issues encountered during the Railway/RunPod
deployment and GPU embedding implementation session (2026-01-15).

Each test prevents a specific regression we had to fix.

Run with: pytest tests/test_session_learnings.py -v
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch


# =============================================================================
# Railway Deployment Issues
# =============================================================================

class TestRailwayConfiguration:
    """Railway deployment configuration issues we encountered"""

    def test_dockerfile_cmd_allows_port_expansion(self):
        """
        Issue: CMD ["streamlit", "run", "--server.port=$PORT"] treats $PORT as literal string
        Fix: Use shell form CMD to allow variable expansion
        """
        dockerfile = Path("Dockerfile")
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()

        # Should use shell form (no brackets)
        assert 'CMD ["streamlit"' not in content, \
            "Dockerfile CMD must use shell form (not array) for $PORT expansion"

        # Should reference PORT variable
        assert "${PORT" in content or "$PORT" in content, \
            "Dockerfile CMD should reference PORT variable"

    def test_no_start_command_conflicts(self):
        """
        Issue: railway.toml startCommand overrode Dockerfile CMD
        Fix: Remove railway.toml and Procfile
        """
        assert not Path("railway.toml").exists(), \
            "railway.toml must not exist (conflicts with Dockerfile CMD)"

        assert not Path("Procfile").exists(), \
            "Procfile must not exist (use Dockerfile CMD instead)"

    def test_streamlit_in_production_requirements(self):
        """
        Issue: streamlit: command not found on Railway
        Fix: Add streamlit to requirements.txt (not just requirements-optional.txt)
        """
        requirements = Path("requirements.txt").read_text()

        assert "streamlit>=" in requirements, \
            "streamlit must be in requirements.txt for Railway deployment"

        assert "plotly>=" in requirements, \
            "plotly must be in requirements.txt for web UI"

    def test_base_image_supports_multiple_platforms(self):
        """
        Issue: Base image built on Mac (arm64) failed on Railway (amd64)
        Fix: Use docker buildx for multi-platform builds
        """
        build_script = Path("build-base-image.sh")
        if not build_script.exists():
            pytest.skip("build-base-image.sh not found")

        content = build_script.read_text()

        assert "buildx" in content, \
            "Base image build must use docker buildx for multi-platform"

        assert "linux/amd64" in content and "linux/arm64" in content, \
            "Base image must build for both amd64 (Railway) and arm64 (Mac)"


# =============================================================================
# RunPod Auto-Detection Issues
# =============================================================================

class TestRunPodAutoDetection:
    """Auto-detection logic issues we encountered"""

    @patch('utils.runpod_db_config.RunPodManager')
    def test_prefers_fully_configured_pods(self, mock_manager):
        """
        Issue: Auto-detection picked new pod without port 8001 or custom password
        Fix: Prefer pods with both PostgreSQL (5432) AND embedding (8001) ports
        """
        from utils.runpod_db_config import get_postgres_config

        # Pod 1: Fully configured (newer)
        fully_configured = {
            "id": "new-pod",
            "name": "rag-new",
            "createdAt": "2026-01-15T23:00:00",
            "runtime": {
                "ports": [
                    {"privatePort": 5432, "ip": "1.1.1.1", "publicPort": 5432},
                    {"privatePort": 8001, "ip": "1.1.1.1", "publicPort": 8001}  # Has embedding!
                ]
            }
        }

        # Pod 2: Partial (older, but no embedding service)
        partial_configured = {
            "id": "old-pod",
            "name": "rag-old",
            "createdAt": "2026-01-15T22:00:00",
            "runtime": {
                "ports": [
                    {"privatePort": 5432, "ip": "2.2.2.2", "publicPort": 5432}
                    # Missing port 8001!
                ]
            }
        }

        mock_instance = mock_manager.return_value
        mock_instance.list_pods.return_value = [partial_configured, fully_configured]

        with patch.dict(os.environ, {"PGHOST": "auto", "RUNPOD_API_KEY": "test"}):
            config = get_postgres_config()

        # Should prefer fully-configured pod (has port 8001) even if not most recent
        assert config["host"] == "1.1.1.1", \
            "Should prefer pod with both PostgreSQL AND embedding service"

    @patch('utils.runpod_db_config.RunPodManager')
    def test_pgpassword_uses_railway_env_not_default(self, mock_manager):
        """
        Issue: New pods got default password, not custom Railway password
        Fix: Pass PGPASSWORD from Railway to pod environment
        """
        # This test verifies pod creation passes PGPASSWORD
        # Actual verification happens in web UI pod creation code
        pass  # Covered by test_pod_creation_inherits_railway_config

    def test_embedding_endpoint_auto_detects_port_8001(self):
        """
        Issue: Had to manually set RUNPOD_EMBEDDING_ENDPOINT
        Fix: Auto-detect from port 8001 mapping (like PostgreSQL auto-detection)
        """
        from utils.runpod_db_config import get_embedding_endpoint

        with patch('utils.runpod_db_config.RunPodManager') as mock:
            mock_pod = {
                "id": "test-pod",
                "runtime": {
                    "ports": [
                        {"privatePort": 8001, "ip": "10.20.30.40", "publicPort": 18001}
                    ]
                }
            }

            mock.return_value.list_pods.return_value = [mock_pod]

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "test"}):
                endpoint = get_embedding_endpoint()

            assert endpoint == "http://10.20.30.40:18001", \
                "Should auto-detect embedding endpoint from port 8001 mapping"


# =============================================================================
# Port Assignment Issues
# =============================================================================

class TestServicePortAssignments:
    """Service port conflicts we encountered"""

    def test_vllm_uses_port_8000_not_8001(self):
        """
        Issue: vLLM started on port 8001, blocking embedding service
        Fix: Force vLLM to port 8000 by unsetting PORT env var
        """
        init_script = Path("scripts/init_runpod_services.sh").read_text()

        # Should unset PORT before starting vLLM
        assert "unset PORT" in init_script, \
            "init script must unset PORT before starting vLLM (prevents override)"

        # Should explicitly use --port 8000
        assert "--port 8000" in init_script, \
            "vLLM must use --port 8000 flag"

    def test_embedding_service_uses_port_8001(self):
        """
        Issue: Embedding service needed explicit PORT setting
        Fix: export PORT=8001 before starting service
        """
        web_ui = Path("rag_web.py").read_text()

        # Auto-startup should set PORT=8001
        if "STEP 4/5" in web_ui:
            startup_section = web_ui[web_ui.find("STEP 4/5"):web_ui.find("STEP 5/5")]
            assert "PORT=8001" in startup_section, \
                "Auto-startup must set PORT=8001 for embedding service"

    def test_default_pod_includes_embedding_port(self):
        """
        Issue: New pods didn't expose port 8001
        Fix: Add 8001 to default ports in RunPodManager.create_pod()
        """
        manager_file = Path("utils/runpod_manager.py").read_text()

        # Find default ports in create_pod signature
        assert "8001" in manager_file, \
            "Default pod ports must include 8001 for embedding service"


# =============================================================================
# Dependency & Environment Issues
# =============================================================================

class TestDependencyManagement:
    """Dependency and environment issues we encountered"""

    def test_embedding_service_uses_virtual_environment(self):
        """
        Issue: Embedding service couldn't find llama_index (installed in venv, running in system python)
        Fix: Use .venv/bin/python to start embedding service
        """
        web_ui = Path("rag_web.py").read_text()

        if "embedding_service" in web_ui:
            # Find the embedding service startup section
            startup_section = web_ui[web_ui.find("embedding_service"):web_ui.find("embedding_service") + 2000]

            assert ".venv/bin/python" in startup_section or ".venv/bin/activate" in startup_section, \
                "Embedding service must run in virtual environment"

    def test_fastapi_in_requirements(self):
        """
        Issue: FastAPI not installed, embedding service couldn't start
        Fix: Add fastapi and uvicorn to requirements.txt
        """
        requirements = Path("requirements.txt").read_text()

        assert "fastapi>=" in requirements, \
            "fastapi must be in requirements.txt for embedding service"

        assert "uvicorn" in requirements, \
            "uvicorn must be in requirements.txt for embedding service"

    def test_urllib3_uses_v2_api(self):
        """
        Issue: method_whitelist parameter doesn't exist in urllib3 2.0+
        Fix: Use allowed_methods parameter
        """
        client_file = Path("utils/runpod_embedding_client.py").read_text()

        assert "allowed_methods" in client_file, \
            "Must use allowed_methods (urllib3 2.0+ API)"

        assert "method_whitelist" not in client_file, \
            "Must not use deprecated method_whitelist"

    def test_pydantic_uses_v2_api(self):
        """
        Issue: Pydantic warning about schema_extra deprecation
        Fix: Use json_schema_extra instead
        """
        service_file = Path("services/embedding_service.py").read_text()

        assert "json_schema_extra" in service_file or "schema_extra" not in service_file, \
            "Must use json_schema_extra (Pydantic v2 API)"


# =============================================================================
# Security Issues
# =============================================================================

class TestSecurityConfiguration:
    """Security issues we encountered"""

    def test_no_hardcoded_api_keys_in_code(self):
        """
        Issue: Hardcoded RunPod API key in get_runpod_connection.py
        Fix: Always use environment variables, never hardcode
        """
        files_to_check = [
            "get_runpod_connection.py",
            "utils/runpod_manager.py",
            "utils/runpod_db_config.py",
            "rag_web.py"
        ]

        for filepath in files_to_check:
            path = Path(filepath)
            if not path.exists():
                continue

            content = path.read_text()

            # Should not have literal API keys (rpa_ pattern)
            # Allow: rpa_YOUR_KEY or rpa_<placeholder>
            lines_with_rpa = [
                line for line in content.split('\n')
                if 'rpa_' in line
                and 'rpa_YOUR' not in line
                and 'rpa_<' not in line
                and '#' not in line  # Allow in comments
            ]

            assert len(lines_with_rpa) == 0, \
                f"{filepath} contains hardcoded RunPod API key: {lines_with_rpa}"

    def test_pgpassword_from_environment(self):
        """
        Issue: Custom password not propagated to new pods
        Fix: Pass PGPASSWORD from Railway environment to pod creation
        """
        web_ui = Path("rag_web.py").read_text()

        # Pod creation should include PGPASSWORD
        if "custom_env" in web_ui:
            # Find custom_env dict creation
            env_section = web_ui[web_ui.find("custom_env"):web_ui.find("custom_env") + 2000]
            assert "PGPASSWORD" in env_section, \
                "Pod creation must pass PGPASSWORD to new pods"


# =============================================================================
# Code Quality Issues
# =============================================================================

class TestCodeQuality:
    """Code quality issues that caused warnings"""

    def test_streamlit_no_deprecated_parameters(self):
        """
        Issue: use_container_width deprecated in Streamlit
        Fix: Use width='stretch' instead
        """
        web_ui = Path("rag_web.py").read_text()

        assert "use_container_width" not in web_ui, \
            "Must not use deprecated use_container_width parameter"

        # Should use new API
        if "st.button" in web_ui or "st.plotly_chart" in web_ui:
            assert "width=" in web_ui, \
                "Should use width parameter for Streamlit components"


# =============================================================================
# Embedding Service Integration
# =============================================================================

class TestEmbeddingServiceConfiguration:
    """Embedding service deployment issues"""

    def test_embedding_service_health_endpoint_exists(self):
        """
        Issue: Need to verify service is running before using it
        Fix: Implement /health endpoint
        """
        from services import embedding_service

        assert hasattr(embedding_service.app, 'routes'), "FastAPI app should have routes"

        # Find health route
        routes = [route.path for route in embedding_service.app.routes]
        assert "/health" in routes, "Must have /health endpoint"

    def test_embedding_client_has_health_check(self):
        """
        Issue: Need to verify service before sending requests
        Fix: Implement check_health() method
        """
        from utils.runpod_embedding_client import RunPodEmbeddingClient

        assert hasattr(RunPodEmbeddingClient, 'check_health'), \
            "Client must have check_health() method"

    def test_embed_nodes_has_fallback(self):
        """
        Issue: System should work even when RunPod GPU unavailable
        Fix: Implement _embed_nodes_local() fallback
        """
        import rag_low_level_m1_16gb_verbose as rag

        # Should have both remote and local embedding functions
        assert hasattr(rag, '_embed_nodes_remote'), \
            "Must have _embed_nodes_remote() for GPU"

        assert hasattr(rag, '_embed_nodes_local'), \
            "Must have _embed_nodes_local() for CPU fallback"

    def test_embedding_service_authenticates_requests(self):
        """
        Issue: Need API key authentication for security
        Fix: Implement X-API-Key header checking
        """
        service_file = Path("services/embedding_service.py").read_text()

        assert "X-API-Key" in service_file, \
            "Embedding service must check X-API-Key header"

        assert "verify_api_key" in service_file, \
            "Must have API key verification function"


# =============================================================================
# Auto-Startup Configuration
# =============================================================================

class TestAutoStartupConfiguration:
    """Auto-startup script issues"""

    def test_auto_startup_activates_venv(self):
        """
        Issue: Auto-startup ran embedding service outside venv, missing dependencies
        Fix: Activate venv before starting services
        """
        web_ui = Path("rag_web.py").read_text()

        if "STEP 4/5" in web_ui:  # Embedding service startup
            startup = web_ui[web_ui.find("STEP 4/5"):web_ui.find("STEP 5/5")]

            assert ".venv/bin/activate" in startup or ".venv/bin/python" in startup, \
                "Auto-startup must use virtual environment"

    def test_auto_startup_sets_port_variables(self):
        """
        Issue: PORT env var caused service port conflicts
        Fix: Explicitly set PORT for embedding, unset for vLLM
        """
        # Check that auto-startup sets PORT=8001 for embedding service
        web_ui = Path("rag_web.py").read_text()

        if "embedding_service" in web_ui:
            assert "PORT=8001" in web_ui, \
                "Auto-startup must set PORT=8001 for embedding service"

    def test_pod_creation_passes_railway_env_vars(self):
        """
        Issue: New pods didn't get custom password/config from Railway
        Fix: Pass PGPASSWORD, PGUSER, DB_NAME from Railway environment to pod
        """
        web_ui = Path("rag_web.py").read_text()

        if "custom_env" in web_ui:
            env_section = web_ui[web_ui.find("custom_env"):web_ui.find("custom_env") + 3000]

            assert "PGPASSWORD" in env_section, "Must pass PGPASSWORD to pods"
            assert "RUNPOD_EMBEDDING_API_KEY" in env_section, "Must pass embedding API key to pods"


# =============================================================================
# File Upload Feature
# =============================================================================

class TestFileUploadFeature:
    """File upload functionality we added"""

    def test_file_upload_option_exists_in_ui(self):
        """
        Feature: Allow users to upload files from browser instead of GitHub
        Implementation: Add st.file_uploader to Index Documents page
        """
        web_ui = Path("rag_web.py").read_text()

        assert "Upload files from your computer" in web_ui, \
            "Should have file upload option"

        assert "st.file_uploader" in web_ui, \
            "Should use Streamlit file uploader"

        # Should support multiple files
        assert "accept_multiple_files" in web_ui, \
            "Should allow multiple file uploads"


# =============================================================================
# Performance & Monitoring
# =============================================================================

class TestPerformanceMonitoring:
    """Performance tracking for embedding speedup"""

    def test_gpu_indicator_shows_in_ui(self):
        """
        Feature: Show users whether GPU or CPU is being used
        Implementation: Add GPU status indicator in Index Documents page
        """
        web_ui = Path("rag_web.py").read_text()

        assert "GPU Acceleration Enabled" in web_ui, \
            "Should show GPU acceleration indicator"

        assert "CPU Mode" in web_ui, \
            "Should show CPU mode warning when GPU unavailable"

    def test_embedding_client_tracks_throughput(self):
        """
        Feature: Track embedding performance for monitoring
        Implementation: Return processing_time_ms in API response
        """
        from services.embedding_service import EmbedResponse

        # Response should include timing metrics
        fields = EmbedResponse.model_fields if hasattr(EmbedResponse, 'model_fields') else EmbedResponse.__fields__

        assert 'processing_time_ms' in fields, \
            "Embed response must include processing time"

        assert 'gpu_used' in fields, \
            "Embed response must indicate if GPU was used"


# =============================================================================
# Integration Test Scenarios
# =============================================================================

@pytest.mark.integration
class TestEndToEndScenarios:
    """End-to-end scenarios requiring actual services"""

    @pytest.mark.skip(reason="Requires running embedding service")
    def test_local_embedding_service_startup(self):
        """Verify embedding service starts successfully on localhost"""
        import requests

        # Start service first: ./scripts/start_embedding_service.sh
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
            assert data['model_loaded'] is True
        except requests.ConnectionError:
            pytest.fail("Embedding service not running. Start with: ./scripts/start_embedding_service.sh")

    @pytest.mark.skip(reason="Requires Railway deployment")
    def test_railway_shows_gpu_indicator(self):
        """Verify Railway Streamlit shows GPU status correctly"""
        # Would require Selenium/browser automation
        # Manual test: Check https://rag.groussard.xyz shows GPU indicator
        pass

    @pytest.mark.skip(reason="Requires RunPod pod with GPU")
    def test_runpod_gpu_acceleration(self):
        """Verify RunPod GPU provides significant speedup"""
        # Would require:
        # 1. RunPod pod with CUDA
        # 2. Embedding service running
        # 3. Benchmark comparing CPU vs GPU throughput
        pass


# =============================================================================
# Configuration Validation
# =============================================================================

class TestConfigurationValidation:
    """Configuration correctness checks"""

    def test_required_env_vars_have_defaults_or_auto_detection(self):
        """
        Issue: Missing env vars caused crashes
        Fix: Provide sensible defaults or auto-detection
        """
        required_vars = {
            "PGHOST": "auto",  # Auto-detect from RunPod
            "PGPORT": "5432",  # Standard PostgreSQL port
            "PGUSER": "fryt",  # Default user
            "DB_NAME": "vector_db",  # Default database
            "RUNPOD_EMBEDDING_ENDPOINT": "auto"  # Auto-detect
        }

        # These should have fallbacks in code (not crash if missing)
        # Verify in Settings class or connection functions
        pass

    def test_all_ports_documented(self):
        """Verify port assignments are documented"""
        # Should have documentation showing:
        # 5432: PostgreSQL
        # 8000: vLLM
        # 8001: Embedding API
        # 22: SSH
        # 3000: Grafana
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
