#!/bin/bash

# Hauptbuch Logs Script
# This script manages and analyzes Hauptbuch logs

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
LOG_TYPE=${1:-all}
LOG_LEVEL=${2:-info}
LINES=${3:-100}
FOLLOW=${4:-false}

# Show logs
show_logs() {
    print_status "Showing Hauptbuch logs..."
    
    case $LOG_TYPE in
        "all")
            print_status "Showing all logs..."
            docker-compose logs --tail=$LINES
            ;;
        "node")
            print_status "Showing Hauptbuch node logs..."
            docker-compose logs --tail=$LINES hauptbuch-node
            ;;
        "redis")
            print_status "Showing Redis logs..."
            docker-compose logs --tail=$LINES hauptbuch-redis
            ;;
        "prometheus")
            print_status "Showing Prometheus logs..."
            docker-compose logs --tail=$LINES hauptbuch-prometheus
            ;;
        "grafana")
            print_status "Showing Grafana logs..."
            docker-compose logs --tail=$LINES hauptbuch-grafana
            ;;
        "system")
            print_status "Showing system logs..."
            journalctl -u hauptbuch --tail=$LINES
            ;;
        "application")
            print_status "Showing application logs..."
            if [ -d "logs" ]; then
                find logs/ -name "*.log" -exec tail -n $LINES {} \;
            else
                print_warning "No application logs found"
            fi
            ;;
        *)
            print_error "Unknown log type: $LOG_TYPE"
            echo "Valid types: all, node, redis, prometheus, grafana, system, application"
            exit 1
            ;;
    esac
}

# Follow logs
follow_logs() {
    print_status "Following Hauptbuch logs..."
    
    case $LOG_TYPE in
        "all")
            print_status "Following all logs..."
            docker-compose logs -f
            ;;
        "node")
            print_status "Following Hauptbuch node logs..."
            docker-compose logs -f hauptbuch-node
            ;;
        "redis")
            print_status "Following Redis logs..."
            docker-compose logs -f hauptbuch-redis
            ;;
        "prometheus")
            print_status "Following Prometheus logs..."
            docker-compose logs -f hauptbuch-prometheus
            ;;
        "grafana")
            print_status "Following Grafana logs..."
            docker-compose logs -f hauptbuch-grafana
            ;;
        "system")
            print_status "Following system logs..."
            journalctl -u hauptbuch -f
            ;;
        "application")
            print_status "Following application logs..."
            if [ -d "logs" ]; then
                tail -f logs/*.log
            else
                print_warning "No application logs found"
            fi
            ;;
        *)
            print_error "Unknown log type: $LOG_TYPE"
            echo "Valid types: all, node, redis, prometheus, grafana, system, application"
            exit 1
            ;;
    esac
}

# Filter logs by level
filter_logs() {
    print_status "Filtering logs by level: $LOG_LEVEL"
    
    case $LOG_LEVEL in
        "error")
            docker-compose logs --tail=$LINES | grep -i "error\|fatal\|panic"
            ;;
        "warn")
            docker-compose logs --tail=$LINES | grep -i "warn\|warning"
            ;;
        "info")
            docker-compose logs --tail=$LINES | grep -i "info"
            ;;
        "debug")
            docker-compose logs --tail=$LINES | grep -i "debug"
            ;;
        "trace")
            docker-compose logs --tail=$LINES | grep -i "trace"
            ;;
        *)
            print_error "Unknown log level: $LOG_LEVEL"
            echo "Valid levels: error, warn, info, debug, trace"
            exit 1
            ;;
    esac
}

# Search logs
search_logs() {
    local search_term=$1
    
    if [ -z "$search_term" ]; then
        print_error "Search term not provided"
        exit 1
    fi
    
    print_status "Searching logs for: $search_term"
    
    case $LOG_TYPE in
        "all")
            docker-compose logs --tail=$LINES | grep -i "$search_term"
            ;;
        "node")
            docker-compose logs --tail=$LINES hauptbuch-node | grep -i "$search_term"
            ;;
        "redis")
            docker-compose logs --tail=$LINES hauptbuch-redis | grep -i "$search_term"
            ;;
        "prometheus")
            docker-compose logs --tail=$LINES hauptbuch-prometheus | grep -i "$search_term"
            ;;
        "grafana")
            docker-compose logs --tail=$LINES hauptbuch-grafana | grep -i "$search_term"
            ;;
        "system")
            journalctl -u hauptbuch --tail=$LINES | grep -i "$search_term"
            ;;
        "application")
            if [ -d "logs" ]; then
                find logs/ -name "*.log" -exec grep -i "$search_term" {} \;
            else
                print_warning "No application logs found"
            fi
            ;;
        *)
            print_error "Unknown log type: $LOG_TYPE"
            echo "Valid types: all, node, redis, prometheus, grafana, system, application"
            exit 1
            ;;
    esac
}

# Analyze logs
analyze_logs() {
    print_status "Analyzing Hauptbuch logs..."
    
    # Error analysis
    local error_count=$(docker-compose logs --tail=1000 | grep -i "error\|fatal\|panic" | wc -l)
    print_status "Error count (last 1000 lines): $error_count"
    
    # Warning analysis
    local warning_count=$(docker-compose logs --tail=1000 | grep -i "warn\|warning" | wc -l)
    print_status "Warning count (last 1000 lines): $warning_count"
    
    # Info analysis
    local info_count=$(docker-compose logs --tail=1000 | grep -i "info" | wc -l)
    print_status "Info count (last 1000 lines): $info_count"
    
    # Debug analysis
    local debug_count=$(docker-compose logs --tail=1000 | grep -i "debug" | wc -l)
    print_status "Debug count (last 1000 lines): $debug_count"
    
    # Trace analysis
    local trace_count=$(docker-compose logs --tail=1000 | grep -i "trace" | wc -l)
    print_status "Trace count (last 1000 lines): $trace_count"
    
    # Performance analysis
    local response_times=$(docker-compose logs --tail=1000 | grep -i "response\|time" | wc -l)
    print_status "Performance-related log entries: $response_times"
    
    # Security analysis
    local security_events=$(docker-compose logs --tail=1000 | grep -i "security\|auth\|login\|logout" | wc -l)
    print_status "Security-related log entries: $security_events"
    
    # Network analysis
    local network_events=$(docker-compose logs --tail=1000 | grep -i "network\|connection\|disconnect" | wc -l)
    print_status "Network-related log entries: $network_events"
}

# Export logs
export_logs() {
    local export_file="logs-export-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Exporting logs to: $export_file"
    
    {
        echo "Hauptbuch Logs Export"
        echo "===================="
        echo "Generated: $(date)"
        echo "Log Type: $LOG_TYPE"
        echo "Log Level: $LOG_LEVEL"
        echo "Lines: $LINES"
        echo ""
        
        case $LOG_TYPE in
            "all")
                docker-compose logs --tail=$LINES
                ;;
            "node")
                docker-compose logs --tail=$LINES hauptbuch-node
                ;;
            "redis")
                docker-compose logs --tail=$LINES hauptbuch-redis
                ;;
            "prometheus")
                docker-compose logs --tail=$LINES hauptbuch-prometheus
                ;;
            "grafana")
                docker-compose logs --tail=$LINES hauptbuch-grafana
                ;;
            "system")
                journalctl -u hauptbuch --tail=$LINES
                ;;
            "application")
                if [ -d "logs" ]; then
                    find logs/ -name "*.log" -exec cat {} \;
                else
                    echo "No application logs found"
                fi
                ;;
        esac
        
    } > "$export_file"
    
    print_success "Logs exported to: $export_file"
}

# Clean logs
clean_logs() {
    print_status "Cleaning Hauptbuch logs..."
    
    # Clean Docker logs
    docker-compose logs --tail=0 > /dev/null 2>&1 || true
    
    # Clean application logs
    if [ -d "logs" ]; then
        find logs/ -name "*.log" -delete
        print_success "Application logs cleaned"
    fi
    
    # Clean system logs
    journalctl --vacuum-time=1d > /dev/null 2>&1 || true
    
    print_success "Logs cleaned successfully"
}

# Rotate logs
rotate_logs() {
    print_status "Rotating Hauptbuch logs..."
    
    # Rotate Docker logs
    docker-compose logs --tail=0 > /dev/null 2>&1 || true
    
    # Rotate application logs
    if [ -d "logs" ]; then
        find logs/ -name "*.log" -exec mv {} {}.old \; 2>/dev/null || true
        print_success "Application logs rotated"
    fi
    
    # Rotate system logs
    journalctl --rotate > /dev/null 2>&1 || true
    
    print_success "Logs rotated successfully"
}

# Show log statistics
show_log_statistics() {
    print_status "Hauptbuch log statistics:"
    echo ""
    
    # Docker logs
    echo "Docker Logs:"
    docker-compose logs --tail=0 > /dev/null 2>&1 || true
    echo "  Total size: $(du -sh /var/lib/docker/containers/ 2>/dev/null | cut -f1 || echo "N/A")"
    echo ""
    
    # Application logs
    if [ -d "logs" ]; then
        echo "Application Logs:"
        echo "  Total size: $(du -sh logs/ | cut -f1)"
        echo "  File count: $(find logs/ -name "*.log" | wc -l)"
        echo ""
    fi
    
    # System logs
    echo "System Logs:"
    echo "  Total size: $(du -sh /var/log/ 2>/dev/null | cut -f1 || echo "N/A")"
    echo "  Journal size: $(journalctl --disk-usage 2>/dev/null | cut -d: -f2 || echo "N/A")"
    echo ""
}

# Main logs function
main() {
    print_status "📋 Starting Hauptbuch logs management..."
    
    if [ "$FOLLOW" = "true" ]; then
        follow_logs
    else
        show_logs
    fi
}

# Handle command line arguments
case "${1:-help}" in
    "all"|"node"|"redis"|"prometheus"|"grafana"|"system"|"application")
        main "$@"
        ;;
    "follow")
        FOLLOW=true
        main "$@"
        ;;
    "filter")
        filter_logs
        ;;
    "search")
        search_logs "$2"
        ;;
    "analyze")
        analyze_logs
        ;;
    "export")
        export_logs
        ;;
    "clean")
        clean_logs
        ;;
    "rotate")
        rotate_logs
        ;;
    "stats")
        show_log_statistics
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Logs Script"
        echo ""
        echo "Usage: $0 [LOG_TYPE] [LOG_LEVEL] [LINES] [FOLLOW]"
        echo ""
        echo "Log Types:"
        echo "  all         - All logs (default)"
        echo "  node        - Hauptbuch node logs"
        echo "  redis       - Redis logs"
        echo "  prometheus  - Prometheus logs"
        echo "  grafana     - Grafana logs"
        echo "  system      - System logs"
        echo "  application - Application logs"
        echo ""
        echo "Commands:"
        echo "  follow      - Follow logs in real-time"
        echo "  filter      - Filter logs by level"
        echo "  search      - Search logs for term"
        echo "  analyze     - Analyze log patterns"
        echo "  export      - Export logs to file"
        echo "  clean       - Clean old logs"
        echo "  rotate      - Rotate log files"
        echo "  stats       - Show log statistics"
        echo "  help        - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  LOG_LEVEL   - Log level: error, warn, info, debug, trace (default: info)"
        echo "  LINES       - Number of lines to show (default: 100)"
        echo "  FOLLOW      - Follow logs in real-time (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
