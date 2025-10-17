#!/bin/bash

# Hauptbuch Cleanup Script
# This script cleans up build artifacts, temporary files, and Docker containers

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
CLEAN_TYPE=${1:-all}
FORCE=${2:-false}

# Clean build artifacts
clean_build() {
    print_status "Cleaning build artifacts..."
    
    # Clean Rust build artifacts
    cargo clean
    
    # Clean target directory
    if [ -d "target" ]; then
        rm -rf target
        print_success "Build artifacts cleaned"
    else
        print_warning "No build artifacts found"
    fi
}

# Clean temporary files
clean_temp() {
    print_status "Cleaning temporary files..."
    
    # Clean temporary files
    find . -name "*.tmp" -delete
    find . -name "*.temp" -delete
    find . -name "*.log" -delete
    find . -name "*.pid" -delete
    
    # Clean backup files
    find . -name "*.bak" -delete
    find . -name "*.backup" -delete
    
    # Clean editor files
    find . -name "*~" -delete
    find . -name "*.swp" -delete
    find . -name "*.swo" -delete
    
    print_success "Temporary files cleaned"
}

# Clean data directories
clean_data() {
    print_status "Cleaning data directories..."
    
    # Clean data directory
    if [ -d "data" ]; then
        if [ "$FORCE" = "true" ]; then
            rm -rf data
            print_success "Data directory cleaned"
        else
            print_warning "Data directory exists. Use --force to remove it"
        fi
    else
        print_warning "No data directory found"
    fi
    
    # Clean logs directory
    if [ -d "logs" ]; then
        rm -rf logs/*
        print_success "Logs directory cleaned"
    else
        print_warning "No logs directory found"
    fi
}

# Clean Docker containers and images
clean_docker() {
    print_status "Cleaning Docker containers and images..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    # Stop and remove containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove Hauptbuch containers
    docker ps -a --filter "name=hauptbuch" --format "{{.ID}}" | xargs -r docker rm -f
    
    # Remove Hauptbuch images
    docker images --filter "reference=hauptbuch*" --format "{{.ID}}" | xargs -r docker rmi -f
    
    # Clean up dangling images
    docker image prune -f
    
    # Clean up dangling volumes
    docker volume prune -f
    
    print_success "Docker containers and images cleaned"
}

# Clean monitoring data
clean_monitoring() {
    print_status "Cleaning monitoring data..."
    
    # Clean Prometheus data
    if [ -d "monitoring/data/prometheus" ]; then
        rm -rf monitoring/data/prometheus/*
        print_success "Prometheus data cleaned"
    fi
    
    # Clean Grafana data
    if [ -d "monitoring/data/grafana" ]; then
        rm -rf monitoring/data/grafana/*
        print_success "Grafana data cleaned"
    fi
    
    # Clean benchmark results
    if [ -d "benchmarks" ]; then
        if [ "$FORCE" = "true" ]; then
            rm -rf benchmarks
            print_success "Benchmark results cleaned"
        else
            print_warning "Benchmark results exist. Use --force to remove them"
        fi
    fi
    
    # Clean test results
    if [ -d "test-results" ]; then
        if [ "$FORCE" = "true" ]; then
            rm -rf test-results
            print_success "Test results cleaned"
        else
            print_warning "Test results exist. Use --force to remove them"
        fi
    fi
}

# Clean cache
clean_cache() {
    print_status "Cleaning cache..."
    
    # Clean Cargo cache
    if [ -d "~/.cargo/registry" ]; then
        rm -rf ~/.cargo/registry/cache
        print_success "Cargo cache cleaned"
    fi
    
    # Clean Rust cache
    if [ -d "~/.rustup" ]; then
        print_warning "Rust toolchain cache exists. Manual cleanup required if needed"
    fi
    
    # Clean system cache
    if [ -d "/tmp" ]; then
        find /tmp -name "*hauptbuch*" -delete 2>/dev/null || true
        print_success "System cache cleaned"
    fi
}

# Clean all
clean_all() {
    print_status "Cleaning everything..."
    
    clean_build
    clean_temp
    clean_data
    clean_docker
    clean_monitoring
    clean_cache
    
    print_success "All cleanup completed!"
}

# Show disk usage
show_disk_usage() {
    print_status "Disk usage before cleanup:"
    df -h .
    
    print_status "Largest directories:"
    du -h . | sort -hr | head -10
}

# Show cleanup summary
show_summary() {
    print_status "Cleanup summary:"
    echo "  - Build artifacts: $(find . -name "target" -type d | wc -l) directories"
    echo "  - Temporary files: $(find . -name "*.tmp" -o -name "*.temp" -o -name "*.log" | wc -l) files"
    echo "  - Data directories: $(find . -name "data" -type d | wc -l) directories"
    echo "  - Docker containers: $(docker ps -a --filter "name=hauptbuch" | wc -l) containers"
    echo "  - Docker images: $(docker images --filter "reference=hauptbuch*" | wc -l) images"
}

# Main cleanup function
main() {
    print_status "🧹 Starting Hauptbuch cleanup..."
    
    show_disk_usage
    
    case $CLEAN_TYPE in
        "build")
            clean_build
            ;;
        "temp")
            clean_temp
            ;;
        "data")
            clean_data
            ;;
        "docker")
            clean_docker
            ;;
        "monitoring")
            clean_monitoring
            ;;
        "cache")
            clean_cache
            ;;
        "all")
            clean_all
            ;;
        *)
            print_error "Unknown clean type: $CLEAN_TYPE"
            echo "Valid types: build, temp, data, docker, monitoring, cache, all"
            exit 1
            ;;
    esac
    
    show_disk_usage
    show_summary
    
    print_success "✅ Cleanup completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "build"|"temp"|"data"|"docker"|"monitoring"|"cache"|"all")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Cleanup Script"
        echo ""
        echo "Usage: $0 [CLEAN_TYPE] [FORCE]"
        echo ""
        echo "Clean Types:"
        echo "  build       - Clean build artifacts"
        echo "  temp        - Clean temporary files"
        echo "  data        - Clean data directories"
        echo "  docker      - Clean Docker containers and images"
        echo "  monitoring  - Clean monitoring data"
        echo "  cache       - Clean cache files"
        echo "  all         - Clean everything (default)"
        echo ""
        echo "Parameters:"
        echo "  FORCE       - Force cleanup of data directories (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
