#!/bin/bash

# Hauptbuch Health Check Script
# This script checks the health of all Hauptbuch services

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

# Check if service is running
check_service() {
    local service_name=$1
    local port=$2
    local url=$3
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        print_success "$service_name is running on port $port"
        return 0
    else
        print_error "$service_name is not responding on port $port"
        return 1
    fi
}

# Check Docker services
check_docker_services() {
    print_status "Checking Docker services..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        return 1
    fi
    
    if ! docker ps > /dev/null 2>&1; then
        print_error "Docker daemon is not running"
        return 1
    fi
    
    # Check if containers are running
    local containers=("hauptbuch-node" "hauptbuch-redis" "hauptbuch-rocksdb" "hauptbuch-prometheus" "hauptbuch-grafana")
    local all_running=true
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "$container"; then
            print_success "Container $container is running"
        else
            print_error "Container $container is not running"
            all_running=false
        fi
    done
    
    if [ "$all_running" = true ]; then
        print_success "All Docker services are running"
        return 0
    else
        print_error "Some Docker services are not running"
        return 1
    fi
}

# Check Hauptbuch node
check_hauptbuch_node() {
    print_status "Checking Hauptbuch node..."
    
    if check_service "Hauptbuch Node" "8545" "http://localhost:8545/health"; then
        return 0
    else
        print_error "Hauptbuch node is not responding"
        return 1
    fi
}

# Check Redis
check_redis() {
    print_status "Checking Redis..."
    
    if check_service "Redis" "6379" "http://localhost:6379"; then
        return 0
    else
        print_error "Redis is not responding"
        return 1
    fi
}

# Check Prometheus
check_prometheus() {
    print_status "Checking Prometheus..."
    
    if check_service "Prometheus" "9090" "http://localhost:9090/-/healthy"; then
        return 0
    else
        print_error "Prometheus is not responding"
        return 1
    fi
}

# Check Grafana
check_grafana() {
    print_status "Checking Grafana..."
    
    if check_service "Grafana" "3000" "http://localhost:3000/api/health"; then
        return 0
    else
        print_error "Grafana is not responding"
        return 1
    fi
}

# Check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
    if (( $(echo "$cpu_usage < 80" | bc -l) )); then
        print_success "CPU usage is normal: ${cpu_usage}%"
    else
        print_warning "CPU usage is high: ${cpu_usage}%"
    fi
    
    # Check memory usage
    local memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage < 80" | bc -l) )); then
        print_success "Memory usage is normal: ${memory_usage}%"
    else
        print_warning "Memory usage is high: ${memory_usage}%"
    fi
    
    # Check disk usage
    local disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [ "$disk_usage" -lt 80 ]; then
        print_success "Disk usage is normal: ${disk_usage}%"
    else
        print_warning "Disk usage is high: ${disk_usage}%"
    fi
}

# Check network connectivity
check_network() {
    print_status "Checking network connectivity..."
    
    # Check if ports are open
    local ports=("30303" "8545" "8546" "6379" "9090" "3000")
    local all_open=true
    
    for port in "${ports[@]}"; do
        if netstat -tuln | grep -q ":$port "; then
            print_success "Port $port is open"
        else
            print_error "Port $port is not open"
            all_open=false
        fi
    done
    
    if [ "$all_open" = true ]; then
        print_success "All required ports are open"
        return 0
    else
        print_error "Some required ports are not open"
        return 1
    fi
}

# Check logs for errors
check_logs() {
    print_status "Checking logs for errors..."
    
    local error_count=0
    
    # Check Docker logs
    if command -v docker &> /dev/null; then
        local containers=("hauptbuch-node" "hauptbuch-redis" "hauptbuch-prometheus" "hauptbuch-grafana")
        for container in "${containers[@]}"; do
            if docker ps --format "table {{.Names}}" | grep -q "$container"; then
                local errors=$(docker logs "$container" 2>&1 | grep -i error | wc -l)
                if [ "$errors" -gt 0 ]; then
                    print_warning "Container $container has $errors errors in logs"
                    error_count=$((error_count + errors))
                else
                    print_success "Container $container has no errors in logs"
                fi
            fi
        done
    fi
    
    if [ "$error_count" -eq 0 ]; then
        print_success "No errors found in logs"
        return 0
    else
        print_warning "Found $error_count errors in logs"
        return 1
    fi
}

# Generate health report
generate_report() {
    local report_file="health-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating health report: $report_file"
    
    {
        echo "Hauptbuch Health Report"
        echo "Generated: $(date)"
        echo "================================"
        echo ""
        
        echo "System Information:"
        echo "OS: $(uname -a)"
        echo "Uptime: $(uptime)"
        echo ""
        
        echo "Docker Services:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo ""
        
        echo "Network Ports:"
        netstat -tuln | grep -E ":(30303|8545|8546|6379|9090|3000)"
        echo ""
        
        echo "System Resources:"
        echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
        echo "Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
        echo "Disk: $(df -h / | awk 'NR==2{print $3 "/" $2 " (" $5 ")"}')"
        echo ""
        
        echo "Service Health:"
        check_hauptbuch_node > /dev/null 2>&1 && echo "Hauptbuch Node: OK" || echo "Hauptbuch Node: FAILED"
        check_redis > /dev/null 2>&1 && echo "Redis: OK" || echo "Redis: FAILED"
        check_prometheus > /dev/null 2>&1 && echo "Prometheus: OK" || echo "Prometheus: FAILED"
        check_grafana > /dev/null 2>&1 && echo "Grafana: OK" || echo "Grafana: FAILED"
        
    } > "$report_file"
    
    print_success "Health report generated: $report_file"
}

# Main health check function
main() {
    print_status "🔍 Starting Hauptbuch health check..."
    
    local overall_status=0
    
    # Run all checks
    check_docker_services || overall_status=1
    check_hauptbuch_node || overall_status=1
    check_redis || overall_status=1
    check_prometheus || overall_status=1
    check_grafana || overall_status=1
    check_system_resources || overall_status=1
    check_network || overall_status=1
    check_logs || overall_status=1
    
    # Generate report
    generate_report
    
    # Final status
    if [ $overall_status -eq 0 ]; then
        print_success "✅ All health checks passed!"
    else
        print_error "❌ Some health checks failed!"
        exit 1
    fi
}

# Run main function
main "$@"
