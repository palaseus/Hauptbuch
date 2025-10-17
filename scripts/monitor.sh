#!/bin/bash

# Hauptbuch Monitoring Script
# This script provides comprehensive monitoring for Hauptbuch

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
MONITOR_TYPE=${1:-all}
INTERVAL=${2:-5}
DURATION=${3:-60}
ALERT_LEVEL=${4:-medium}

# Monitoring results
MONITORING_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Add monitoring result
add_result() {
    local check_name=$1
    local status=$2
    local message=$3
    local value=$4
    
    MONITORING_RESULTS+=("$check_name:$status:$message:$value")
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

# Monitor system metrics
monitor_system_metrics() {
    print_status "Monitoring system metrics..."
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    if (( $(echo "$cpu_usage < 80" | bc -l) )); then
        add_result "CPU Usage" "PASS" "CPU usage is normal" "${cpu_usage}%"
    else
        add_result "CPU Usage" "FAIL" "CPU usage is high" "${cpu_usage}%"
    fi
    
    # Memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage < 80" | bc -l) )); then
        add_result "Memory Usage" "PASS" "Memory usage is normal" "${memory_usage}%"
    else
        add_result "Memory Usage" "FAIL" "Memory usage is high" "${memory_usage}%"
    fi
    
    # Disk usage
    local disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        add_result "Disk Usage" "PASS" "Disk usage is normal" "${disk_usage}%"
    else
        add_result "Disk Usage" "FAIL" "Disk usage is high" "${disk_usage}%"
    fi
    
    # Load average
    local load_avg=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}' || echo "0")
    local cpu_cores=$(nproc 2>/dev/null || echo "1")
    if (( $(echo "$load_avg < $cpu_cores" | bc -l) )); then
        add_result "Load Average" "PASS" "Load average is normal" "$load_avg"
    else
        add_result "Load Average" "FAIL" "Load average is high" "$load_avg"
    fi
}

# Monitor Docker metrics
monitor_docker_metrics() {
    print_status "Monitoring Docker metrics..."
    
    if ! command -v docker &> /dev/null; then
        add_result "Docker Status" "WARN" "Docker not installed" "N/A"
        return 0
    fi
    
    # Docker daemon status
    if docker ps > /dev/null 2>&1; then
        add_result "Docker Daemon" "PASS" "Docker daemon is running" "OK"
    else
        add_result "Docker Daemon" "FAIL" "Docker daemon is not running" "N/A"
    fi
    
    # Container status
    local running_containers=$(docker ps --filter "name=hauptbuch" --format "{{.Names}}" | wc -l)
    if [ "$running_containers" -gt 0 ]; then
        add_result "Container Status" "PASS" "Hauptbuch containers are running" "$running_containers"
    else
        add_result "Container Status" "FAIL" "No Hauptbuch containers running" "0"
    fi
    
    # Container resource usage
    local container_memory=$(docker stats --no-stream --format "table {{.MemUsage}}" | grep hauptbuch | awk '{print $1}' | head -1)
    if [ -n "$container_memory" ]; then
        add_result "Container Memory" "PASS" "Container memory usage is reasonable" "$container_memory"
    else
        add_result "Container Memory" "WARN" "Container memory usage not available" "N/A"
    fi
}

# Monitor network metrics
monitor_network_metrics() {
    print_status "Monitoring network metrics..."
    
    # Network connectivity
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        add_result "Network Connectivity" "PASS" "Network is accessible" "OK"
    else
        add_result "Network Connectivity" "FAIL" "Network is not accessible" "N/A"
    fi
    
    # Network latency
    local latency=$(ping -c 1 8.8.8.8 2>/dev/null | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}' || echo "0")
    if (( $(echo "$latency < 100" | bc -l) )); then
        add_result "Network Latency" "PASS" "Network latency is good" "${latency}ms"
    else
        add_result "Network Latency" "FAIL" "Network latency is high" "${latency}ms"
    fi
    
    # Listening ports
    local open_ports=$(netstat -tuln 2>/dev/null | grep LISTEN | wc -l)
    if [ "$open_ports" -le 20 ]; then
        add_result "Open Ports" "PASS" "Reasonable number of open ports" "$open_ports"
    else
        add_result "Open Ports" "FAIL" "Too many open ports" "$open_ports"
    fi
}

# Monitor service metrics
monitor_service_metrics() {
    print_status "Monitoring service metrics..."
    
    # Hauptbuch node
    if curl -s http://localhost:8545/health > /dev/null 2>&1; then
        add_result "Hauptbuch Node" "PASS" "Hauptbuch node is responding" "OK"
    else
        add_result "Hauptbuch Node" "FAIL" "Hauptbuch node is not responding" "N/A"
    fi
    
    # Redis
    if curl -s http://localhost:6379 > /dev/null 2>&1; then
        add_result "Redis" "PASS" "Redis is responding" "OK"
    else
        add_result "Redis" "FAIL" "Redis is not responding" "N/A"
    fi
    
    # Prometheus
    if curl -s http://localhost:9090/-/healthy > /dev/null 2>&1; then
        add_result "Prometheus" "PASS" "Prometheus is responding" "OK"
    else
        add_result "Prometheus" "FAIL" "Prometheus is not responding" "N/A"
    fi
    
    # Grafana
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        add_result "Grafana" "PASS" "Grafana is responding" "OK"
    else
        add_result "Grafana" "FAIL" "Grafana is not responding" "N/A"
    fi
}

# Monitor application metrics
monitor_application_metrics() {
    print_status "Monitoring application metrics..."
    
    # Build status
    if cargo check --quiet 2>/dev/null; then
        add_result "Build Status" "PASS" "Project builds successfully" "OK"
    else
        add_result "Build Status" "FAIL" "Project build failed" "N/A"
    fi
    
    # Test status
    if cargo test --quiet 2>/dev/null; then
        add_result "Test Status" "PASS" "All tests pass" "OK"
    else
        add_result "Test Status" "FAIL" "Some tests fail" "N/A"
    fi
    
    # Code quality
    if cargo clippy --quiet 2>/dev/null; then
        add_result "Code Quality" "PASS" "Clippy checks pass" "OK"
    else
        add_result "Code Quality" "FAIL" "Clippy warnings found" "N/A"
    fi
}

# Monitor all metrics
monitor_all_metrics() {
    print_status "🔍 Starting comprehensive monitoring..."
    
    monitor_system_metrics
    monitor_docker_metrics
    monitor_network_metrics
    monitor_service_metrics
    monitor_application_metrics
    
    print_status "Monitoring completed!"
}

# Continuous monitoring
continuous_monitoring() {
    print_status "Starting continuous monitoring..."
    print_status "Monitoring interval: ${INTERVAL}s"
    print_status "Monitoring duration: ${DURATION}s"
    print_status "Alert level: $ALERT_LEVEL"
    echo ""
    
    local start_time=$(date +%s)
    local end_time=$((start_time + DURATION))
    
    while [ $(date +%s) -lt $end_time ]; do
        print_status "Monitoring cycle: $(date)"
        
        # Clear previous results
        MONITORING_RESULTS=()
        TOTAL_CHECKS=0
        PASSED_CHECKS=0
        FAILED_CHECKS=0
        
        # Run monitoring
        monitor_all_metrics
        
        # Check for alerts
        if [ "$FAILED_CHECKS" -gt 0 ]; then
            print_error "⚠️  $FAILED_CHECKS issues detected!"
            
            if [ "$ALERT_LEVEL" = "high" ]; then
                # Send alert (implement your alerting mechanism)
                print_error "🚨 HIGH ALERT: Critical issues detected!"
            fi
        fi
        
        # Wait for next cycle
        sleep $INTERVAL
    done
    
    print_success "Continuous monitoring completed!"
}

# Generate monitoring report
generate_report() {
    local report_file="monitoring-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating monitoring report: $report_file"
    
    {
        echo "Hauptbuch Monitoring Report"
        echo "=========================="
        echo "Generated: $(date)"
        echo "Interval: ${INTERVAL}s"
        echo "Duration: ${DURATION}s"
        echo "Alert Level: $ALERT_LEVEL"
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
        for result in "${MONITORING_RESULTS[@]}"; do
            IFS=':' read -r name status message value <<< "$result"
            echo "$name: $status - $message ($value)"
        done
        
    } > "$report_file"
    
    print_success "Monitoring report generated: $report_file"
}

# Show monitoring summary
show_summary() {
    print_status "Monitoring Summary:"
    echo "  Total Checks: $TOTAL_CHECKS"
    echo "  Passed: $PASSED_CHECKS"
    echo "  Failed: $FAILED_CHECKS"
    echo "  Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
    echo ""
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        print_success "✅ All monitoring checks passed!"
    else
        print_error "❌ $FAILED_CHECKS monitoring checks failed!"
    fi
}

# Main monitoring function
main() {
    print_status "🔍 Starting Hauptbuch monitoring..."
    
    case $MONITOR_TYPE in
        "system")
            monitor_system_metrics
            ;;
        "docker")
            monitor_docker_metrics
            ;;
        "network")
            monitor_network_metrics
            ;;
        "services")
            monitor_service_metrics
            ;;
        "application")
            monitor_application_metrics
            ;;
        "all")
            monitor_all_metrics
            ;;
        "continuous")
            continuous_monitoring
            ;;
        *)
            print_error "Unknown monitoring type: $MONITOR_TYPE"
            echo "Valid types: system, docker, network, services, application, all, continuous"
            exit 1
            ;;
    esac
    
    show_summary
    
    if [ "$ALERT_LEVEL" = "high" ]; then
        generate_report
    fi
}

# Handle command line arguments
case "${1:-help}" in
    "system"|"docker"|"network"|"services"|"application"|"all"|"continuous")
        main "$@"
        ;;
    "report")
        generate_report
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Monitoring Script"
        echo ""
        echo "Usage: $0 [MONITOR_TYPE] [INTERVAL] [DURATION] [ALERT_LEVEL]"
        echo ""
        echo "Monitoring Types:"
        echo "  system       - System metrics"
        echo "  docker       - Docker metrics"
        echo "  network      - Network metrics"
        echo "  services     - Service metrics"
        echo "  application  - Application metrics"
        echo "  all          - All metrics (default)"
        echo "  continuous   - Continuous monitoring"
        echo ""
        echo "Commands:"
        echo "  report       - Generate monitoring report"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  INTERVAL     - Monitoring interval in seconds (default: 5)"
        echo "  DURATION      - Monitoring duration in seconds (default: 60)"
        echo "  ALERT_LEVEL  - Alert level: low, medium, high (default: medium)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
