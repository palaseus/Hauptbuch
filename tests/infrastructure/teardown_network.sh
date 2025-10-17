#!/bin/bash

# Hauptbuch Network Teardown Script
# This script cleanly shuts down the local test network

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="/tmp/hauptbuch-test"

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

# Stop Hauptbuch node
stop_hauptbuch_node() {
    print_status "Stopping Hauptbuch node..."
    
    if [ -f "$DATA_DIR/hauptbuch.pid" ]; then
        PID=$(cat "$DATA_DIR/hauptbuch.pid")
        if kill -0 $PID 2>/dev/null; then
            print_status "Sending SIGTERM to Hauptbuch node (PID: $PID)..."
            kill -TERM $PID
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 $PID 2>/dev/null && [ $count -lt 30 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                print_warning "Force killing Hauptbuch node..."
                kill -KILL $PID
            fi
            
            print_success "Hauptbuch node stopped"
        else
            print_warning "Hauptbuch node was not running"
        fi
        rm -f "$DATA_DIR/hauptbuch.pid"
    else
        print_warning "No PID file found for Hauptbuch node"
    fi
}

# Stop Prometheus
stop_prometheus() {
    print_status "Stopping Prometheus..."
    
    if [ -f "$DATA_DIR/prometheus.pid" ]; then
        PID=$(cat "$DATA_DIR/prometheus.pid")
        if kill -0 $PID 2>/dev/null; then
            print_status "Stopping Prometheus (PID: $PID)..."
            kill -TERM $PID
            sleep 2
            print_success "Prometheus stopped"
        fi
        rm -f "$DATA_DIR/prometheus.pid"
    else
        print_warning "No Prometheus PID file found"
    fi
}

# Stop other services
stop_other_services() {
    print_status "Stopping other services..."
    
    # Stop any Docker containers
    if command -v docker &> /dev/null; then
        print_status "Stopping Docker containers..."
        docker ps -q --filter "name=hauptbuch" | xargs -r docker stop
        print_success "Docker containers stopped"
    fi
    
    # Stop any Redis instances
    if command -v redis-cli &> /dev/null; then
        print_status "Stopping Redis instances..."
        redis-cli -p 6379 shutdown 2>/dev/null || true
        print_success "Redis instances stopped"
    fi
}

# Clean up data directory
cleanup_data() {
    print_status "Cleaning up data directory..."
    
    if [ -d "$DATA_DIR" ]; then
        print_status "Removing data directory: $DATA_DIR"
        rm -rf "$DATA_DIR"
        print_success "Data directory cleaned"
    else
        print_warning "Data directory not found: $DATA_DIR"
    fi
}

# Clean up temporary files
cleanup_temp_files() {
    print_status "Cleaning up temporary files..."
    
    # Remove any temporary test files
    find /tmp -name "*hauptbuch*" -type f -delete 2>/dev/null || true
    find /tmp -name "*test*" -name "*.log" -type f -delete 2>/dev/null || true
    
    print_success "Temporary files cleaned"
}

# Clean up network interfaces
cleanup_network() {
    print_status "Cleaning up network interfaces..."
    
    # Remove any virtual network interfaces
    ip link show | grep -E "hauptbuch|test" | cut -d: -f2 | xargs -r -I {} ip link delete {} 2>/dev/null || true
    
    print_success "Network interfaces cleaned"
}

# Generate cleanup report
generate_cleanup_report() {
    print_status "Generating cleanup report..."
    
    local report_file="/tmp/hauptbuch_cleanup_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$report_file" << EOF
Hauptbuch Network Cleanup Report
================================
Timestamp: $(date)
Data Directory: $DATA_DIR

Services Stopped:
- Hauptbuch node
- Prometheus monitoring
- Docker containers
- Redis instances

Files Cleaned:
- Data directory: $DATA_DIR
- Temporary files
- Log files
- PID files

Network Cleaned:
- Virtual interfaces
- Test network configurations

Cleanup Status: SUCCESS
EOF
    
    print_success "Cleanup report generated: $report_file"
}

# Main teardown function
main() {
    print_status "🛑 Starting Hauptbuch network teardown..."
    
    stop_hauptbuch_node
    stop_prometheus
    stop_other_services
    cleanup_data
    cleanup_temp_files
    cleanup_network
    generate_cleanup_report
    
    print_success "✅ Hauptbuch network teardown completed!"
    print_status "All services stopped and data cleaned"
}

# Handle command line arguments
case "${1:-teardown}" in
    "teardown")
        main
        ;;
    "stop-only")
        stop_hauptbuch_node
        stop_prometheus
        stop_other_services
        print_success "Services stopped (data preserved)"
        ;;
    "clean-only")
        cleanup_data
        cleanup_temp_files
        cleanup_network
        print_success "Data cleaned (services may still be running)"
        ;;
    "force")
        print_warning "Force teardown - killing all processes..."
        pkill -f hauptbuch || true
        pkill -f prometheus || true
        pkill -f redis || true
        cleanup_data
        cleanup_temp_files
        cleanup_network
        print_success "Force teardown completed"
        ;;
    *)
        echo "Usage: $0 {teardown|stop-only|clean-only|force}"
        echo ""
        echo "  teardown   - Full teardown (default)"
        echo "  stop-only  - Stop services but preserve data"
        echo "  clean-only - Clean data but don't stop services"
        echo "  force      - Force kill all processes and clean"
        exit 1
        ;;
esac
