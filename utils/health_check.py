"""
health_check.py - Comprehensive health check system for RAG pipeline

Provides:
  - Database connectivity checks
  - Model availability checks
  - System resource checks
  - Dependency verification
  - Health endpoint for monitoring

Usage:
    from utils.health_check import HealthChecker

    checker = HealthChecker()
    status = checker.check_all()
    print(status)
"""

import os
import sys
import time
import psutil
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import psycopg2
from psycopg2 import OperationalError as PgOperationalError

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        return {
            "component": self.component,
            "status": self.status,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
            "timestamp": self.timestamp
        }


class HealthChecker:
    """Comprehensive health checker for RAG pipeline"""

    def __init__(self):
        self.checks: List[HealthCheckResult] = []

    def check_database(self) -> HealthCheckResult:
        """Check PostgreSQL database connectivity and health"""
        start_time = time.time()

        try:
            # Get connection parameters
            conn_params = {
                "host": os.getenv("PGHOST", "localhost"),
                "port": int(os.getenv("PGPORT", "5432")),
                "user": os.getenv("PGUSER", "postgres"),
                "password": os.getenv("PGPASSWORD", ""),
                "dbname": os.getenv("DB_NAME", "vector_db"),
            }

            # Test connection
            conn = psycopg2.connect(**conn_params)
            conn.autocommit = True
            cursor = conn.cursor()

            # Check database version
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # Check pgvector extension
            cursor.execute("SELECT extname FROM pg_extension WHERE extname='vector';")
            has_pgvector = cursor.fetchone() is not None

            # Check table count
            cursor.execute("""
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema='public' AND table_name LIKE 'data_%';
            """)
            table_count = cursor.fetchone()[0]

            # Check database size
            cursor.execute(f"""
                SELECT pg_size_pretty(pg_database_size('{conn_params["dbname"]}'));
            """)
            db_size = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component="database",
                status="healthy",
                message="Database connection successful",
                latency_ms=latency_ms,
                details={
                    "host": conn_params["host"],
                    "port": conn_params["port"],
                    "database": conn_params["dbname"],
                    "pgvector_enabled": has_pgvector,
                    "vector_tables": table_count,
                    "size": db_size,
                    "version": version.split(",")[0]
                }
            )

        except PgOperationalError as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="database",
                status="unhealthy",
                message=f"Database connection failed: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)}
            )
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="database",
                status="unhealthy",
                message=f"Database check failed: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)}
            )

    def check_system_resources(self) -> HealthCheckResult:
        """Check system resource availability"""
        start_time = time.time()

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Determine status
            status = "healthy"
            warnings = []

            if cpu_percent > 85:
                status = "degraded"
                warnings.append(f"High CPU usage: {cpu_percent}%")

            if memory_percent > 90:
                status = "degraded"
                warnings.append(f"High memory usage: {memory_percent}%")

            if disk_percent > 85:
                status = "degraded"
                warnings.append(f"High disk usage: {disk_percent}%")

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                component="system_resources",
                status=status,
                message="System resources OK" if status == "healthy" else "; ".join(warnings),
                latency_ms=latency_ms,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3)
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="system_resources",
                status="unhealthy",
                message=f"Resource check failed: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)}
            )

    def check_gpu_availability(self) -> HealthCheckResult:
        """Check GPU/accelerator availability"""
        start_time = time.time()

        if not TORCH_AVAILABLE:
            return HealthCheckResult(
                component="gpu",
                status="degraded",
                message="PyTorch not available",
                latency_ms=(time.time() - start_time) * 1000,
                details={"torch_available": False}
            )

        try:
            # Check CUDA
            cuda_available = torch.cuda.is_available()
            cuda_devices = torch.cuda.device_count() if cuda_available else 0

            # Check MPS (Apple Silicon)
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

            # Determine status
            if cuda_available or mps_available:
                status = "healthy"
                message = f"GPU available: {'CUDA' if cuda_available else 'MPS'}"
            else:
                status = "degraded"
                message = "No GPU available, using CPU"

            latency_ms = (time.time() - start_time) * 1000

            details = {
                "torch_available": True,
                "cuda_available": cuda_available,
                "mps_available": mps_available,
                "device_count": cuda_devices
            }

            if cuda_available:
                details["cuda_version"] = torch.version.cuda
                details["devices"] = [torch.cuda.get_device_name(i) for i in range(cuda_devices)]

            return HealthCheckResult(
                component="gpu",
                status=status,
                message=message,
                latency_ms=latency_ms,
                details=details
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component="gpu",
                status="degraded",
                message=f"GPU check failed: {str(e)}",
                latency_ms=latency_ms,
                details={"error": str(e)}
            )

    def check_dependencies(self) -> HealthCheckResult:
        """Check critical dependencies"""
        start_time = time.time()

        required_packages = [
            "llama_index",
            "psycopg2",
            "torch",
            "transformers",
        ]

        optional_packages = [
            "vllm",
            "sentence_transformers",
            "tqdm",
        ]

        missing_required = []
        missing_optional = []

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)

        for package in optional_packages:
            try:
                __import__(package)
            except ImportError:
                missing_optional.append(package)

        latency_ms = (time.time() - start_time) * 1000

        if missing_required:
            return HealthCheckResult(
                component="dependencies",
                status="unhealthy",
                message=f"Missing required packages: {', '.join(missing_required)}",
                latency_ms=latency_ms,
                details={
                    "missing_required": missing_required,
                    "missing_optional": missing_optional
                }
            )
        elif missing_optional:
            return HealthCheckResult(
                component="dependencies",
                status="degraded",
                message=f"Missing optional packages: {', '.join(missing_optional)}",
                latency_ms=latency_ms,
                details={
                    "missing_required": [],
                    "missing_optional": missing_optional
                }
            )
        else:
            return HealthCheckResult(
                component="dependencies",
                status="healthy",
                message="All dependencies available",
                latency_ms=latency_ms,
                details={
                    "missing_required": [],
                    "missing_optional": []
                }
            )

    def check_models(self) -> HealthCheckResult:
        """Check model files availability"""
        start_time = time.time()

        # Check for LLM model
        llm_model_path = os.getenv("MODEL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
        llm_exists = os.path.exists(llm_model_path)

        # Check embedding model (typically downloaded to cache)
        embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en")

        latency_ms = (time.time() - start_time) * 1000

        details = {
            "llm_model_path": llm_model_path,
            "llm_model_exists": llm_exists,
            "embedding_model": embed_model
        }

        if llm_exists:
            details["llm_model_size_gb"] = os.path.getsize(llm_model_path) / (1024**3)

        if llm_exists:
            status = "healthy"
            message = "Models available"
        else:
            status = "degraded"
            message = f"LLM model not found at {llm_model_path}"

        return HealthCheckResult(
            component="models",
            status=status,
            message=message,
            latency_ms=latency_ms,
            details=details
        )

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks"""
        start_time = time.time()

        self.checks = [
            self.check_database(),
            self.check_system_resources(),
            self.check_gpu_availability(),
            self.check_dependencies(),
            self.check_models(),
        ]

        total_latency_ms = (time.time() - start_time) * 1000

        # Determine overall status
        if any(c.status == "unhealthy" for c in self.checks):
            overall_status = "unhealthy"
        elif any(c.status == "degraded" for c in self.checks):
            overall_status = "degraded"
        else:
            overall_status = "healthy"

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "total_latency_ms": total_latency_ms,
            "checks": [c.to_dict() for c in self.checks],
            "summary": {
                "healthy": sum(1 for c in self.checks if c.status == "healthy"),
                "degraded": sum(1 for c in self.checks if c.status == "degraded"),
                "unhealthy": sum(1 for c in self.checks if c.status == "unhealthy"),
                "total": len(self.checks)
            }
        }

    def get_readiness(self) -> Dict[str, Any]:
        """Check if system is ready to serve requests"""
        db_check = self.check_database()
        deps_check = self.check_dependencies()

        ready = (
            db_check.status in ["healthy", "degraded"] and
            deps_check.status in ["healthy", "degraded"]
        )

        return {
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "database": db_check.to_dict(),
                "dependencies": deps_check.to_dict()
            }
        }

    def get_liveness(self) -> Dict[str, Any]:
        """Check if application is alive (basic check)"""
        return {
            "alive": True,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - psutil.boot_time()
        }


# CLI for running health checks
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)

    checker = HealthChecker()

    if len(sys.argv) > 1:
        check_type = sys.argv[1]

        if check_type == "readiness":
            result = checker.get_readiness()
        elif check_type == "liveness":
            result = checker.get_liveness()
        else:
            result = checker.check_all()
    else:
        result = checker.check_all()

    print(json.dumps(result, indent=2))

    # Exit with appropriate code
    sys.exit(0 if result.get("status") == "healthy" or result.get("ready") else 1)
