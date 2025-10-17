#!/bin/bash

# Hauptbuch Testing Script
# This script runs comprehensive tests for Hauptbuch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Configuration
TEST_TYPE=${1:-all}
COVERAGE=${2:-false}
PARALLEL=${3:-true}
RESULTS_DIR="test-results/$(date +%Y%m%d-%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run unit tests
run_unit_tests() {
    print_status "Running unit tests..."
    
    local output_file="$RESULTS_DIR/unit-tests.txt"
    
    {
        echo "Unit Tests"
        echo "=========="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        if [ "$COVERAGE" = "true" ]; then
            cargo tarpaulin --out Html --output-dir "$RESULTS_DIR" -- --test-threads=1
        else
            cargo test --lib -- --nocapture
        fi
        
    } > "$output_file"
    
    print_success "Unit tests completed: $output_file"
}

# Run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    local output_file="$RESULTS_DIR/integration-tests.txt"
    
    {
        echo "Integration Tests"
        echo "================="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        cargo test --test integration -- --nocapture
        
    } > "$output_file"
    
    print_success "Integration tests completed: $output_file"
}

# Run performance tests
run_performance_tests() {
    print_status "Running performance tests..."
    
    local output_file="$RESULTS_DIR/performance-tests.txt"
    
    {
        echo "Performance Tests"
        echo "================="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        cargo test --test performance -- --nocapture
        
    } > "$output_file"
    
    print_success "Performance tests completed: $output_file"
}

# Run security tests
run_security_tests() {
    print_status "Running security tests..."
    
    local output_file="$RESULTS_DIR/security-tests.txt"
    
    {
        echo "Security Tests"
        echo "=============="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        cargo test --test security -- --nocapture
        
    } > "$output_file"
    
    print_success "Security tests completed: $output_file"
}

# Run chaos engineering tests
run_chaos_tests() {
    print_status "Running chaos engineering tests..."
    
    local output_file="$RESULTS_DIR/chaos-tests.txt"
    
    {
        echo "Chaos Engineering Tests"
        echo "======================="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        cargo test --test chaos_engineering -- --nocapture
        
    } > "$output_file"
    
    print_success "Chaos engineering tests completed: $output_file"
}

# Run fuzzing tests
run_fuzzing_tests() {
    print_status "Running fuzzing tests..."
    
    local output_file="$RESULTS_DIR/fuzzing-tests.txt"
    
    {
        echo "Fuzzing Tests"
        echo "============="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        cargo test --test fuzzing -- --nocapture
        
    } > "$output_file"
    
    print_success "Fuzzing tests completed: $output_file"
}

# Run property-based tests
run_property_tests() {
    print_status "Running property-based tests..."
    
    local output_file="$RESULTS_DIR/property-tests.txt"
    
    {
        echo "Property-Based Tests"
        echo "===================="
        echo "Timestamp: $(date)"
        echo "Coverage: $COVERAGE"
        echo "Parallel: $PARALLEL"
        echo ""
        
        cargo test --test property -- --nocapture
        
    } > "$output_file"
    
    print_success "Property-based tests completed: $output_file"
}

# Run all tests
run_all_tests() {
    print_status "Running all tests..."
    
    run_unit_tests
    run_integration_tests
    run_performance_tests
    run_security_tests
    run_chaos_tests
    run_fuzzing_tests
    run_property_tests
    
    print_success "All tests completed!"
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

# Run test coverage
run_coverage() {
    print_status "Running test coverage..."
    
    local output_file="$RESULTS_DIR/coverage-report.html"
    
    cargo tarpaulin --out Html --output-dir "$RESULTS_DIR" -- --test-threads=1
    
    print_success "Test coverage completed: $output_file"
}

# Run test analysis
run_analysis() {
    print_status "Running test analysis..."
    
    local output_file="$RESULTS_DIR/test-analysis.txt"
    
    {
        echo "Test Analysis"
        echo "============="
        echo "Timestamp: $(date)"
        echo ""
        
        echo "Test Statistics:"
        echo "Total tests: $(find . -name "*.rs" -exec grep -l "#\[test\]" {} \; | wc -l)"
        echo "Integration tests: $(find . -name "integration*.rs" | wc -l)"
        echo "Performance tests: $(find . -name "performance*.rs" | wc -l)"
        echo "Security tests: $(find . -name "security*.rs" | wc -l)"
        echo ""
        
        echo "Code Coverage:"
        if [ -f "$RESULTS_DIR/coverage-report.html" ]; then
            echo "Coverage report generated"
        else
            echo "No coverage report available"
        fi
        
    } > "$output_file"
    
    print_success "Test analysis completed: $output_file"
}

# Main testing function
main() {
    print_status "🧪 Starting Hauptbuch tests..."
    
    case $TEST_TYPE in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "security")
            run_security_tests
            ;;
        "chaos")
            run_chaos_tests
            ;;
        "fuzzing")
            run_fuzzing_tests
            ;;
        "property")
            run_property_tests
            ;;
        "coverage")
            run_coverage
            ;;
        "analysis")
            run_analysis
            ;;
        "all")
            run_all_tests
            ;;
        *)
            print_error "Unknown test type: $TEST_TYPE"
            echo "Valid types: unit, integration, performance, security, chaos, fuzzing, property, coverage, analysis, all"
            exit 1
            ;;
    esac
    
    generate_report
    run_analysis
    
    print_success "✅ Tests completed successfully!"
    print_status "Results saved to: $RESULTS_DIR"
}

# Handle command line arguments
case "${1:-help}" in
    "unit"|"integration"|"performance"|"security"|"chaos"|"fuzzing"|"property"|"coverage"|"analysis"|"all")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Testing Script"
        echo ""
        echo "Usage: $0 [TEST_TYPE] [COVERAGE] [PARALLEL]"
        echo ""
        echo "Test Types:"
        echo "  unit         - Unit tests"
        echo "  integration  - Integration tests"
        echo "  performance  - Performance tests"
        echo "  security     - Security tests"
        echo "  chaos        - Chaos engineering tests"
        echo "  fuzzing      - Fuzzing tests"
        echo "  property     - Property-based tests"
        echo "  coverage     - Test coverage analysis"
        echo "  analysis     - Test analysis"
        echo "  all          - Run all tests (default)"
        echo ""
        echo "Parameters:"
        echo "  COVERAGE     - Enable test coverage (default: false)"
        echo "  PARALLEL     - Enable parallel execution (default: true)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
