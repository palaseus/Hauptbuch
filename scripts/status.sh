#!/bin/bash

# Hauptbuch Status Script
# This script shows the current status of Hauptbuch services

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
STATUS_TYPE=${1:-all}
VERBOSE=${2:-false}

# Check service status
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

# Show system information
show_system_info() {
    print_status "System Information:"
    echo "  OS: $(uname -a)"
    echo "  Uptime: $(uptime)"
    echo "  Load: $(cat /proc/loadavg 2>/dev/null || echo "N/A")"
    echo "  Memory: $(free -h | grep Mem | awk '{print $3 "/" $2 " (" $3/$2*100 "%)"}')"
    echo "  Disk: $(df -h / | awk 'NR==2{print $3 "/" $2 " (" $5 ")"}')"
    echo "  CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
    echo ""
}

# Show Docker status
show_docker_status() {
    print_status "Docker Status:"
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    if ! docker ps > /dev/null 2>&1; then
        print_error "Docker daemon is not running"
        return 1
    fi
    
    echo "  Docker version: $(docker --version)"
    echo "  Docker Compose version: $(docker-compose --version)"
    echo ""
    
    echo "  Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep hauptbuch || echo "    No Hauptbuch containers running"
    echo ""
    
    echo "  Images:"
    docker images --filter "reference=hauptbuch*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" || echo "    No Hauptbuch images found"
    echo ""
}

# Show network status
show_network_status() {
    print_status "Network Status:"
    
    echo "  Listening ports:"
    netstat -tuln | grep -E ":(30303|8545|8546|6379|9090|3000)" || echo "    No Hauptbuch ports listening"
    echo ""
    
    echo "  Network interfaces:"
    ip addr show | grep -E "inet " | grep -v "127.0.0.1" || echo "    No network interfaces found"
    echo ""
}

# Show service status
show_service_status() {
    print_status "Service Status:"
    
    # Check Hauptbuch node
    if check_service "Hauptbuch Node" "8545" "http://localhost:8545/health"; then
        echo "    RPC: http://localhost:8545"
        echo "    WebSocket: ws://localhost:8546"
    fi
    
    # Check Redis
    if check_service "Redis" "6379" "http://localhost:6379"; then
        echo "    Redis: redis://localhost:6379"
    fi
    
    # Check Prometheus
    if check_service "Prometheus" "9090" "http://localhost:9090/-/healthy"; then
        echo "    Prometheus: http://localhost:9090"
    fi
    
    # Check Grafana
    if check_service "Grafana" "3000" "http://localhost:3000/api/health"; then
        echo "    Grafana: http://localhost:3000"
    fi
    
    echo ""
}

# Show project status
show_project_status() {
    print_status "Project Status:"
    
    echo "  Rust version: $(rustc --version)"
    echo "  Cargo version: $(cargo --version)"
    echo ""
    
    if [ -d ".git" ]; then
        echo "  Git status:"
        echo "    Branch: $(git branch --show-current)"
        echo "    Status: $(git status --porcelain | wc -l) changes"
        echo "    Last commit: $(git log -1 --format="%h %s" 2>/dev/null || echo "No commits")"
        echo ""
    fi
    
    echo "  Build status:"
    if cargo check --quiet 2>/dev/null; then
        print_success "    Build: OK"
    else
        print_error "    Build: FAILED"
    fi
    
    echo "  Test status:"
    if cargo test --quiet 2>/dev/null; then
        print_success "    Tests: OK"
    else
        print_error "    Tests: FAILED"
    fi
    
    echo ""
}

# Show configuration status
show_config_status() {
    print_status "Configuration Status:"
    
    echo "  Configuration files:"
    [ -f "config.toml" ] && print_success "    config.toml: OK" || print_error "    config.toml: MISSING"
    [ -f ".env" ] && print_success "    .env: OK" || print_warning "    .env: MISSING"
    [ -f "docker-compose.yml" ] && print_success "    docker-compose.yml: OK" || print_error "    docker-compose.yml: MISSING"
    echo ""
    
    echo "  Data directories:"
    [ -d "data" ] && print_success "    data/: OK" || print_warning "    data/: MISSING"
    [ -d "logs" ] && print_success "    logs/: OK" || print_warning "    logs/: MISSING"
    [ -d "monitoring" ] && print_success "    monitoring/: OK" || print_warning "    monitoring/: MISSING"
    echo ""
}

# Show monitoring status
show_monitoring_status() {
    print_status "Monitoring Status:"
    
    echo "  Prometheus:"
    if [ -d "monitoring/data/prometheus" ]; then
        echo "    Data directory: OK"
        echo "    Size: $(du -sh monitoring/data/prometheus 2>/dev/null | cut -f1 || echo "N/A")"
    else
        print_warning "    Data directory: MISSING"
    fi
    
    echo "  Grafana:"
    if [ -d "monitoring/data/grafana" ]; then
        echo "    Data directory: OK"
        echo "    Size: $(du -sh monitoring/data/grafana 2>/dev/null | cut -f1 || echo "N/A")"
    else
        print_warning "    Data directory: MISSING"
    fi
    
    echo ""
}

# Show performance status
show_performance_status() {
    print_status "Performance Status:"
    
    echo "  CPU usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')%"
    echo "  Memory usage: $(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')%"
    echo "  Disk usage: $(df -h / | awk 'NR==2{print $5}' | sed 's/%//')%"
    echo ""
    
    echo "  Load average: $(cat /proc/loadavg 2>/dev/null || echo "N/A")"
    echo "  Processes: $(ps aux | wc -l) total"
    echo ""
}

# Show security status
show_security_status() {
    print_status "Security Status:"
    
    echo "  Firewall:"
    if command -v ufw &> /dev/null; then
        ufw status | grep -q "Status: active" && print_success "    UFW: ACTIVE" || print_warning "    UFW: INACTIVE"
    else
        print_warning "    UFW: NOT INSTALLED"
    fi
    
    echo "  SSL certificates:"
    if [ -f "ssl/cert.pem" ]; then
        print_success "    SSL: OK"
    else
        print_warning "    SSL: MISSING"
    fi
    
    echo "  Security updates:"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        local updates=$(apt list --upgradable 2>/dev/null | wc -l)
        echo "    Available updates: $updates"
    fi
    
    echo ""
}

# Show all status
show_all_status() {
    print_status "🔍 Hauptbuch Status Report"
    echo "================================"
    echo "Generated: $(date)"
    echo ""
    
    show_system_info
    show_docker_status
    show_network_status
    show_service_status
    show_project_status
    show_config_status
    show_monitoring_status
    show_performance_status
    show_security_status
    
    print_success "✅ Status report completed!"
}

# Generate status report
generate_report() {
    local report_file="status-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating status report: $report_file"
    
    {
        echo "Hauptbuch Status Report"
        echo "======================="
        echo "Generated: $(date)"
        echo ""
        
        show_system_info
        show_docker_status
        show_network_status
        show_service_status
        show_project_status
        show_config_status
        show_monitoring_status
        show_performance_status
        show_security_status
        
    } > "$report_file"
    
    print_success "Status report generated: $report_file"
}

# Main status function
main() {
    print_status "🔍 Starting Hauptbuch status check..."
    
    case $STATUS_TYPE in
        "system")
            show_system_info
            ;;
        "docker")
            show_docker_status
            ;;
        "network")
            show_network_status
            ;;
        "services")
            show_service_status
            ;;
        "project")
            show_project_status
            ;;
        "config")
            show_config_status
            ;;
        "monitoring")
            show_monitoring_status
            ;;
        "performance")
            show_performance_status
            ;;
        "security")
            show_security_status
            ;;
        "all")
            show_all_status
            ;;
        *)
            print_error "Unknown status type: $STATUS_TYPE"
            echo "Valid types: system, docker, network, services, project, config, monitoring, performance, security, all"
            exit 1
            ;;
    esac
    
    if [ "$VERBOSE" = "true" ]; then
        generate_report
    fi
}

# Handle command line arguments
case "${1:-help}" in
    "system"|"docker"|"network"|"services"|"project"|"config"|"monitoring"|"performance"|"security"|"all")
        main "$@"
        ;;
    "report")
        generate_report
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Status Script"
        echo ""
        echo "Usage: $0 [STATUS_TYPE] [VERBOSE]"
        echo ""
        echo "Status Types:"
        echo "  system       - System information"
        echo "  docker       - Docker status"
        echo "  network      - Network status"
        echo "  services     - Service status"
        echo "  project      - Project status"
        echo "  config       - Configuration status"
        echo "  monitoring   - Monitoring status"
        echo "  performance  - Performance status"
        echo "  security     - Security status"
        echo "  all          - All status (default)"
        echo ""
        echo "Commands:"
        echo "  report       - Generate status report"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  VERBOSE      - Generate detailed report (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
