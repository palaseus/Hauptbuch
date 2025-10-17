#!/bin/bash

# Hauptbuch Comprehensive Test Runner
# This script runs all tests in the Hauptbuch test suite

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="test-results/$(date +%Y%m%d-%H%M%S)"
PARALLEL=${PARALLEL:-true}
COVERAGE=${COVERAGE:-false}
VERBOSE=${VERBOSE:-false}

# Test categories
CATEGORIES=("infrastructure" "integration" "contracts" "api" "performance" "security")
ALL_TESTS=false

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Create results directory
mkdir -p "$RESULTS_DIR"

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --category)
                CATEGORY="$2"
                shift 2
                ;;
            --parallel)
                PARALLEL=true
                shift
                ;;
            --no-parallel)
                PARALLEL=false
                shift
                ;;
            --coverage)
                COVERAGE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --all)
                ALL_TESTS=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

show_help() {
    echo "Hauptbuch Test Runner"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --category CATEGORY    Run specific test category"
    echo "  --parallel            Run tests in parallel (default)"
    echo "  --no-parallel         Run tests sequentially"
    echo "  --coverage            Generate test coverage report"
    echo "  --verbose             Verbose output"
    echo "  --all                 Run all test categories"
    echo "  --help                Show this help message"
    echo ""
    echo "Test Categories:"
    echo "  infrastructure        Network setup and deployment tests"
    echo "  integration          Blockchain functionality tests"
    echo "  contracts            Smart contract tests"
    echo "  api                  RPC/API interaction tests"
    echo "  performance          Performance benchmarks"
    echo "  security             Security validation tests"
    echo ""
    echo "Examples:"
    echo "  $0 --category integration"
    echo "  $0 --category contracts --coverage"
    echo "  $0 --all --parallel"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3."
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 not found. Please install pip3."
        exit 1
    fi
    
    # Check pytest
    if ! python3 -c "import pytest" &> /dev/null; then
        print_warning "pytest not found. Installing..."
        pip3 install pytest pytest-asyncio pytest-cov
    fi
    
    # Check Node.js for contract tests
    if ! command -v node &> /dev/null; then
        print_warning "Node.js not found. Contract tests may not work."
    fi
    
    # Check Rust for building
    if ! command -v cargo &> /dev/null; then
        print_error "Rust/Cargo not found. Please install Rust."
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "$SCRIPT_DIR/utils/requirements.txt" ]; then
        pip3 install -r "$SCRIPT_DIR/utils/requirements.txt"
        print_success "Python dependencies installed"
    else
        print_warning "requirements.txt not found"
    fi
}

# Run infrastructure tests
run_infrastructure_tests() {
    print_status "Running infrastructure tests..."
    
    local output_file="$RESULTS_DIR/infrastructure-tests.txt"
    
    {
        echo "Infrastructure Tests"
        echo "==================="
        echo "Timestamp: $(date)"
        echo ""
        
        # Test network setup
        if [ -f "$SCRIPT_DIR/infrastructure/setup_local_network.sh" ]; then
            echo "Testing network setup..."
            "$SCRIPT_DIR/infrastructure/setup_local_network.sh" health
        fi
        
        # Test contract deployment
        if [ -f "$SCRIPT_DIR/infrastructure/deploy_contracts.sh" ]; then
            echo "Testing contract deployment..."
            "$SCRIPT_DIR/infrastructure/deploy_contracts.sh" status
        fi
        
    } > "$output_file"
    
    print_success "Infrastructure tests completed: $output_file"
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    local output_file="$RESULTS_DIR/integration-tests.txt"
    local test_dir="$SCRIPT_DIR/integration"
    
    {
        echo "Integration Tests"
        echo "================"
        echo "Timestamp: $(date)"
        echo ""
        
        # Run Python tests
        if [ -d "$test_dir" ]; then
            cd "$test_dir"
            python3 -m pytest -v --tb=short
        fi
        
    } > "$output_file"
    
    print_success "Integration tests completed: $output_file"
}

# Run contract tests
run_contract_tests() {
    print_status "Running contract tests..."
    
    local output_file="$RESULTS_DIR/contract-tests.txt"
    local contract_dir="$SCRIPT_DIR/contracts"
    
    {
        echo "Contract Tests"
        echo "=============="
        echo "Timestamp: $(date)"
        echo ""
        
        if [ -d "$contract_dir" ]; then
            cd "$contract_dir"
            
            # Install Node.js dependencies
            if [ -f "package.json" ]; then
                npm install
            fi
            
            # Run Hardhat tests
            if [ -f "hardhat.config.js" ]; then
                npx hardhat test
            fi
            
            # Run coverage if requested
            if [ "$COVERAGE" = "true" ]; then
                npx hardhat coverage
            fi
        fi
        
    } > "$output_file"
    
    print_success "Contract tests completed: $output_file"
}

# Run API tests
run_api_tests() {
    print_status "Running API tests..."
    
    local output_file="$RESULTS_DIR/api-tests.txt"
    local api_dir="$SCRIPT_DIR/api"
    
    {
        echo "API Tests"
        echo "========="
        echo "Timestamp: $(date)"
        echo ""
        
        if [ -d "$api_dir" ]; then
            cd "$api_dir"
            python3 -m pytest -v --tb=short
        fi
        
    } > "$output_file"
    
    print_success "API tests completed: $output_file"
}

# Run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    
    local output_file="$RESULTS_DIR/performance-tests.txt"
    local perf_dir="$SCRIPT_DIR/performance"
    
    {
        echo "Performance Tests"
        echo "================="
        echo "Timestamp: $(date)"
        echo ""
        
        if [ -d "$perf_dir" ]; then
            cd "$perf_dir"
            python3 -m pytest -v --tb=short
        fi
        
    } > "$output_file"
    
    print_success "Performance tests completed: $output_file"
}

# Run security tests
run_security_tests() {
    print_status "Running security tests..."
    
    local output_file="$RESULTS_DIR/security-tests.txt"
    local security_dir="$SCRIPT_DIR/security"
    
    {
        echo "Security Tests"
        echo "=============="
        echo "Timestamp: $(date)"
        echo ""
        
        if [ -d "$security_dir" ]; then
            cd "$security_dir"
            python3 -m pytest -v --tb=short
        fi
        
    } > "$output_file"
    
    print_success "Security tests completed: $output_file"
}

# Generate test report
generate_report() {
    print_status "Generating test report..."
    
    local report_file="$RESULTS_DIR/test-report.html"
    
    {
        echo "<!DOCTYPE html>"
        echo "<html>"
        echo "<head>"
        echo "  <title>Hauptbuch Test Report</title>"
        echo "  <style>"
        echo "    body { font-family: Arial, sans-serif; margin: 20px; }"
        echo "    h1, h2 { color: #333; }"
        echo "    .success { color: green; }"
        echo "    .warning { color: orange; }"
        echo "    .error { color: red; }"
        echo "    pre { background: #f5f5f5; padding: 10px; border-radius: 5px; }"
        echo "    table { border-collapse: collapse; width: 100%; }"
        echo "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }"
        echo "    th { background-color: #f2f2f2; }"
        echo "  </style>"
        echo "</head>"
        echo "<body>"
        echo "  <h1>Hauptbuch Test Report</h1>"
        echo "  <p><strong>Generated:</strong> $(date)</p>"
        echo "  <p><strong>Environment:</strong> $(uname -a)</p>"
        echo "  <p><strong>Coverage:</strong> $COVERAGE</p>"
        echo "  <p><strong>Parallel:</strong> $PARALLEL</p>"
        echo ""
        echo "  <h2>System Information</h2>"
        echo "  <pre>"
        echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
        echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
        echo "Disk: $(df -h / | awk 'NR==2{print $2}')"
        echo "  </pre>"
        echo ""
        echo "  <h2>Test Results</h2>"
        
        # Include all test results
        for file in "$RESULTS_DIR"/*.txt; do
            if [ -f "$file" ]; then
                echo "  <h3>$(basename "$file" .txt)</h3>"
                echo "  <pre>"
                cat "$file"
                echo "  </pre>"
            fi
        done
        
        echo "</body>"
        echo "</html>"
        
    } > "$report_file"
    
    print_success "Test report generated: $report_file"
}

# Main test runner
main() {
    print_status "🧪 Starting Hauptbuch test suite..."
    
    check_prerequisites
    install_dependencies
    
    if [ "$ALL_TESTS" = "true" ]; then
        run_infrastructure_tests
        run_integration_tests
        run_contract_tests
        run_api_tests
        run_performance_tests
        run_security_tests
    elif [ -n "$CATEGORY" ]; then
        case "$CATEGORY" in
            "infrastructure")
                run_infrastructure_tests
                ;;
            "integration")
                run_integration_tests
                ;;
            "contracts")
                run_contract_tests
                ;;
            "api")
                run_api_tests
                ;;
            "performance")
                run_performance_tests
                ;;
            "security")
                run_security_tests
                ;;
            *)
                print_error "Unknown category: $CATEGORY"
                show_help
                exit 1
                ;;
        esac
    else
        print_error "No test category specified. Use --category or --all"
        show_help
        exit 1
    fi
    
    generate_report
    
    print_success "✅ Test suite completed!"
    print_status "Results saved to: $RESULTS_DIR"
}

# Parse arguments and run
parse_arguments "$@"
main
