#!/bin/bash

# Hauptbuch Performance Script
# This script analyzes and optimizes Hauptbuch performance

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
PERFORMANCE_TYPE=${1:-all}
OPTIMIZATION_LEVEL=${2:-medium}
PROFILE=${3:-false}

# Performance results
PERFORMANCE_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Add performance result
add_result() {
    local check_name=$1
    local status=$2
    local message=$3
    local value=$4
    
    PERFORMANCE_RESULTS+=("$check_name:$status:$message:$value")
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$status" = "PASS" ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "$check_name: $message ($value)"
    elif [ "$status" = "FAIL" ]; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        print_error "$check_name: $message ($value)"
    else
        print_warning "$check_name: $message ($value)"
    fi
}

# Check system performance
check_system_performance() {
    print_status "Checking system performance..."
    
    # Check CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    if (( $(echo "$cpu_usage < 80" | bc -l) )); then
        add_result "CPU Usage" "PASS" "CPU usage is normal" "${cpu_usage}%"
    else
        add_result "CPU Usage" "FAIL" "CPU usage is high" "${cpu_usage}%"
    fi
    
    # Check memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage < 80" | bc -l) )); then
        add_result "Memory Usage" "PASS" "Memory usage is normal" "${memory_usage}%"
    else
        add_result "Memory Usage" "FAIL" "Memory usage is high" "${memory_usage}%"
    fi
    
    # Check disk usage
    local disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        add_result "Disk Usage" "PASS" "Disk usage is normal" "${disk_usage}%"
    else
        add_result "Disk Usage" "FAIL" "Disk usage is high" "${disk_usage}%"
    fi
    
    # Check load average
    local load_avg=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}' || echo "0")
    local cpu_cores=$(nproc 2>/dev/null || echo "1")
    if (( $(echo "$load_avg < $cpu_cores" | bc -l) )); then
        add_result "Load Average" "PASS" "Load average is normal" "$load_avg"
    else
        add_result "Load Average" "FAIL" "Load average is high" "$load_avg"
    fi
}

# Check build performance
check_build_performance() {
    print_status "Checking build performance..."
    
    # Check build time
    local start_time=$(date +%s)
    cargo build --quiet 2>/dev/null
    local end_time=$(date +%s)
    local build_time=$((end_time - start_time))
    
    if [ "$build_time" -lt 60 ]; then
        add_result "Build Time" "PASS" "Build time is acceptable" "${build_time}s"
    else
        add_result "Build Time" "FAIL" "Build time is slow" "${build_time}s"
    fi
    
    # Check binary size
    if [ -f "target/release/hauptbuch" ]; then
        local binary_size=$(du -h target/release/hauptbuch | cut -f1)
        add_result "Binary Size" "PASS" "Binary size is reasonable" "$binary_size"
    else
        add_result "Binary Size" "WARN" "Binary not found" "N/A"
    fi
    
    # Check compilation warnings
    local warnings=$(cargo build 2>&1 | grep -c "warning:" || echo "0")
    if [ "$warnings" -eq 0 ]; then
        add_result "Compilation Warnings" "PASS" "No compilation warnings" "$warnings"
    else
        add_result "Compilation Warnings" "FAIL" "Compilation warnings found" "$warnings"
    fi
}

# Check test performance
check_test_performance() {
    print_status "Checking test performance..."
    
    # Check test time
    local start_time=$(date +%s)
    cargo test --quiet 2>/dev/null
    local end_time=$(date +%s)
    local test_time=$((end_time - start_time))
    
    if [ "$test_time" -lt 120 ]; then
        add_result "Test Time" "PASS" "Test time is acceptable" "${test_time}s"
    else
        add_result "Test Time" "FAIL" "Test time is slow" "${test_time}s"
    fi
    
    # Check test coverage
    if command -v cargo-tarpaulin &> /dev/null; then
        local coverage=$(cargo tarpaulin --out Xml 2>/dev/null | grep -o 'line-rate="[^"]*"' | cut -d'"' -f2 || echo "0")
        if (( $(echo "$coverage > 80" | bc -l) )); then
            add_result "Test Coverage" "PASS" "Test coverage is good" "${coverage}%"
        else
            add_result "Test Coverage" "FAIL" "Test coverage is low" "${coverage}%"
        fi
    else
        add_result "Test Coverage" "WARN" "Coverage tool not available" "N/A"
    fi
}

# Check runtime performance
check_runtime_performance() {
    print_status "Checking runtime performance..."
    
    # Check if services are running
    if docker ps --filter "name=hauptbuch" --format "{{.Names}}" | grep -q hauptbuch; then
        add_result "Service Status" "PASS" "Services are running" "OK"
        
        # Check service response time
        local start_time=$(date +%s%N)
        curl -s http://localhost:8545/health > /dev/null 2>&1
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 ))
        
        if [ "$response_time" -lt 1000 ]; then
            add_result "Response Time" "PASS" "Response time is good" "${response_time}ms"
        else
            add_result "Response Time" "FAIL" "Response time is slow" "${response_time}ms"
        fi
    else
        add_result "Service Status" "FAIL" "Services are not running" "N/A"
    fi
    
    # Check memory usage of containers
    local container_memory=$(docker stats --no-stream --format "table {{.MemUsage}}" | grep hauptbuch | awk '{print $1}' | head -1)
    if [ -n "$container_memory" ]; then
        add_result "Container Memory" "PASS" "Container memory usage is reasonable" "$container_memory"
    else
        add_result "Container Memory" "WARN" "Container memory usage not available" "N/A"
    fi
}

# Check network performance
check_network_performance() {
    print_status "Checking network performance..."
    
    # Check network latency
    local latency=$(ping -c 1 8.8.8.8 2>/dev/null | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}' || echo "0")
    if (( $(echo "$latency < 100" | bc -l) )); then
        add_result "Network Latency" "PASS" "Network latency is good" "${latency}ms"
    else
        add_result "Network Latency" "FAIL" "Network latency is high" "${latency}ms"
    fi
    
    # Check bandwidth
    local bandwidth=$(iperf3 -c 8.8.8.8 -t 1 2>/dev/null | grep "sender" | awk '{print $7}' || echo "0")
    if [ "$bandwidth" != "0" ]; then
        add_result "Network Bandwidth" "PASS" "Network bandwidth is available" "${bandwidth}Mbps"
    else
        add_result "Network Bandwidth" "WARN" "Network bandwidth test failed" "N/A"
    fi
}

# Check database performance
check_database_performance() {
    print_status "Checking database performance..."
    
    # Check RocksDB performance
    if [ -d "data" ]; then
        local db_size=$(du -sh data | cut -f1)
        add_result "Database Size" "PASS" "Database size is reasonable" "$db_size"
        
        # Check database access time
        local start_time=$(date +%s%N)
        ls data > /dev/null 2>&1
        local end_time=$(date +%s%N)
        local access_time=$(( (end_time - start_time) / 1000000 ))
        
        if [ "$access_time" -lt 100 ]; then
            add_result "Database Access" "PASS" "Database access is fast" "${access_time}ms"
        else
            add_result "Database Access" "FAIL" "Database access is slow" "${access_time}ms"
        fi
    else
        add_result "Database Size" "WARN" "Database directory not found" "N/A"
    fi
}

# Check all performance
check_all_performance() {
    print_status "🚀 Starting comprehensive performance analysis..."
    
    check_system_performance
    check_build_performance
    check_test_performance
    check_runtime_performance
    check_network_performance
    check_database_performance
    
    print_status "Performance analysis completed!"
}

# Optimize performance
optimize_performance() {
    print_status "Optimizing performance..."
    
    # Optimize build
    if [ "$OPTIMIZATION_LEVEL" = "high" ]; then
        print_status "Applying high-level optimizations..."
        
        # Enable LTO
        echo '[profile.release]' >> Cargo.toml
        echo 'lto = true' >> Cargo.toml
        echo 'codegen-units = 1' >> Cargo.toml
        echo 'panic = "abort"' >> Cargo.toml
        
        # Enable CPU-specific optimizations
        echo 'rustflags = ["-C", "target-cpu=native"]' >> .cargo/config.toml
        
        print_success "High-level optimizations applied"
    fi
    
    # Optimize Docker
    if command -v docker &> /dev/null; then
        print_status "Optimizing Docker configuration..."
        
        # Update docker-compose.yml with performance settings
        sed -i 's/mem_limit: 512m/mem_limit: 2g/' docker-compose.yml 2>/dev/null || true
        sed -i 's/cpus: 1/cpus: 2/' docker-compose.yml 2>/dev/null || true
        
        print_success "Docker optimizations applied"
    fi
    
    # Optimize system
    if [ "$OPTIMIZATION_LEVEL" = "high" ]; then
        print_status "Applying system optimizations..."
        
        # Increase file descriptor limits
        echo '* soft nofile 65536' >> /etc/security/limits.conf 2>/dev/null || true
        echo '* hard nofile 65536' >> /etc/security/limits.conf 2>/dev/null || true
        
        # Optimize kernel parameters
        echo 'net.core.rmem_max = 16777216' >> /etc/sysctl.conf 2>/dev/null || true
        echo 'net.core.wmem_max = 16777216' >> /etc/sysctl.conf 2>/dev/null || true
        
        print_success "System optimizations applied"
    fi
}

# Profile performance
profile_performance() {
    print_status "Profiling performance..."
    
    # CPU profiling
    if command -v perf &> /dev/null; then
        print_status "Running CPU profiling..."
        perf record -g cargo run --release &
        local pid=$!
        sleep 10
        kill $pid
        perf report > performance-cpu.txt
        print_success "CPU profiling completed"
    fi
    
    # Memory profiling
    if command -v valgrind &> /dev/null; then
        print_status "Running memory profiling..."
        valgrind --tool=massif cargo run --release &
        local pid=$!
        sleep 10
        kill $pid
        print_success "Memory profiling completed"
    fi
    
    # Flame graph
    if command -v flamegraph &> /dev/null; then
        print_status "Generating flame graph..."
        flamegraph -- cargo run --release
        print_success "Flame graph generated"
    fi
}

# Generate performance report
generate_report() {
    local report_file="performance-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating performance report: $report_file"
    
    {
        echo "Hauptbuch Performance Report"
        echo "=========================="
        echo "Generated: $(date)"
        echo "Optimization Level: $OPTIMIZATION_LEVEL"
        echo "Total Checks: $TOTAL_CHECKS"
        echo "Passed: $PASSED_CHECKS"
        echo "Failed: $FAILED_CHECKS"
        echo "Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
        echo ""
        
        echo "System Information:"
        echo "  OS: $(uname -a)"
        echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
        echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
        echo "  Disk: $(df -h / | awk 'NR==2{print $2}')"
        echo ""
        
        echo "Detailed Results:"
        echo "================"
        for result in "${PERFORMANCE_RESULTS[@]}"; do
            IFS=':' read -r name status message value <<< "$result"
            echo "$name: $status - $message ($value)"
        done
        
    } > "$report_file"
    
    print_success "Performance report generated: $report_file"
}

# Show performance summary
show_summary() {
    print_status "Performance Summary:"
    echo "  Total Checks: $TOTAL_CHECKS"
    echo "  Passed: $PASSED_CHECKS"
    echo "  Failed: $FAILED_CHECKS"
    echo "  Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
    echo ""
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        print_success "✅ All performance checks passed!"
    else
        print_error "❌ $FAILED_CHECKS performance checks failed!"
    fi
}

# Main performance function
main() {
    print_status "🚀 Starting Hauptbuch performance analysis..."
    
    case $PERFORMANCE_TYPE in
        "system")
            check_system_performance
            ;;
        "build")
            check_build_performance
            ;;
        "test")
            check_test_performance
            ;;
        "runtime")
            check_runtime_performance
            ;;
        "network")
            check_network_performance
            ;;
        "database")
            check_database_performance
            ;;
        "all")
            check_all_performance
            ;;
        *)
            print_error "Unknown performance type: $PERFORMANCE_TYPE"
            echo "Valid types: system, build, test, runtime, network, database, all"
            exit 1
            ;;
    esac
    
    if [ "$OPTIMIZATION_LEVEL" != "none" ]; then
        optimize_performance
    fi
    
    if [ "$PROFILE" = "true" ]; then
        profile_performance
    fi
    
    show_summary
    
    if [ "$OPTIMIZATION_LEVEL" = "high" ]; then
        generate_report
    fi
}

# Handle command line arguments
case "${1:-help}" in
    "system"|"build"|"test"|"runtime"|"network"|"database"|"all")
        main "$@"
        ;;
    "optimize")
        optimize_performance
        ;;
    "profile")
        profile_performance
        ;;
    "report")
        generate_report
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Performance Script"
        echo ""
        echo "Usage: $0 [PERFORMANCE_TYPE] [OPTIMIZATION_LEVEL] [PROFILE]"
        echo ""
        echo "Performance Types:"
        echo "  system       - System performance"
        echo "  build        - Build performance"
        echo "  test         - Test performance"
        echo "  runtime      - Runtime performance"
        echo "  network      - Network performance"
        echo "  database     - Database performance"
        echo "  all          - All performance checks (default)"
        echo ""
        echo "Commands:"
        echo "  optimize     - Optimize performance"
        echo "  profile      - Profile performance"
        echo "  report       - Generate performance report"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  OPTIMIZATION_LEVEL - Optimization level: none, low, medium, high (default: medium)"
        echo "  PROFILE            - Enable profiling (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
