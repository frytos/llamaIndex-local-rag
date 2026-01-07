#!/bin/bash
# Performance Regression Test Runner
# Usage: ./run_performance_tests.sh [mode]
# Modes: fast, slow, all, regression, benchmark

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default mode
MODE=${1:-fast}

echo -e "${BLUE}=== Performance Regression Test Suite ===${NC}"
echo -e "${BLUE}Mode: ${MODE}${NC}\n"

case "$MODE" in
  fast)
    echo -e "${GREEN}Running fast tests (no @pytest.mark.slow)...${NC}"
    python -m pytest tests/test_performance_regression.py -v -m "not slow" --tb=short
    ;;

  slow)
    echo -e "${YELLOW}Running slow tests only...${NC}"
    python -m pytest tests/test_performance_regression.py -v -m slow --tb=short
    ;;

  all)
    echo -e "${GREEN}Running all tests...${NC}"
    python -m pytest tests/test_performance_regression.py -v --tb=short
    ;;

  regression)
    echo -e "${GREEN}Running regression detection tests only...${NC}"
    python -m pytest tests/test_performance_regression.py::TestRegressionDetection -v --tb=short
    ;;

  benchmark)
    echo -e "${YELLOW}Running benchmark integration tests...${NC}"
    python -m pytest tests/test_performance_regression.py::TestBenchmarkIntegration -v --tb=short
    ;;

  memory)
    echo -e "${GREEN}Running memory performance tests...${NC}"
    python -m pytest tests/test_performance_regression.py::TestMemoryPerformance -v --tb=short
    ;;

  config)
    echo -e "${GREEN}Running configuration performance tests...${NC}"
    python -m pytest tests/test_performance_regression.py::TestConfigurationPerformance -v --tb=short
    ;;

  help)
    echo "Performance Test Runner"
    echo ""
    echo "Usage: ./run_performance_tests.sh [mode]"
    echo ""
    echo "Available modes:"
    echo "  fast        - Run fast tests only (default, ~1.5s)"
    echo "  slow        - Run slow tests only (~6s)"
    echo "  all         - Run all tests (~8s)"
    echo "  regression  - Run regression detection tests only"
    echo "  benchmark   - Run benchmark integration tests"
    echo "  memory      - Run memory performance tests"
    echo "  config      - Run configuration performance tests"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_performance_tests.sh fast"
    echo "  ./run_performance_tests.sh all"
    echo "  ./run_performance_tests.sh regression"
    ;;

  *)
    echo -e "${RED}Unknown mode: ${MODE}${NC}"
    echo "Run './run_performance_tests.sh help' for usage information"
    exit 1
    ;;
esac

exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo -e "\n${GREEN}✓ All tests passed!${NC}"
else
  echo -e "\n${RED}✗ Some tests failed${NC}"
fi

exit $exit_code
