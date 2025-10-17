#!/bin/bash

# Hauptbuch Development Script
# This script provides common development tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to show help
show_help() {
    echo "Hauptbuch Development Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the project"
    echo "  test        Run tests"
    echo "  bench       Run benchmarks"
    echo "  fmt         Format code"
    echo "  clippy      Run clippy linter"
    echo "  audit       Check for security vulnerabilities"
    echo "  clean       Clean build artifacts"
    echo "  dev         Start development server with hot reload"
    echo "  docker      Start Docker services"
    echo "  stop        Stop Docker services"
    echo "  logs        Show Docker logs"
    echo "  monitor     Open monitoring dashboard"
    echo "  help        Show this help message"
}

# Function to build the project
build() {
    print_status "Building Hauptbuch..."
    cargo build
    print_success "Build completed!"
}

# Function to run tests
test() {
    print_status "Running tests..."
    cargo test
    print_success "Tests completed!"
}

# Function to run benchmarks
bench() {
    print_status "Running benchmarks..."
    cargo bench
    print_success "Benchmarks completed!"
}

# Function to format code
fmt() {
    print_status "Formatting code..."
    cargo fmt
    print_success "Code formatted!"
}

# Function to run clippy
clippy() {
    print_status "Running clippy..."
    cargo clippy -- -D warnings
    print_success "Clippy completed!"
}

# Function to run audit
audit() {
    print_status "Checking for security vulnerabilities..."
    cargo audit
    print_success "Security audit completed!"
}

# Function to clean build artifacts
clean() {
    print_status "Cleaning build artifacts..."
    cargo clean
    print_success "Clean completed!"
}

# Function to start development server
dev() {
    print_status "Starting development server with hot reload..."
    cargo watch -x run
}

# Function to start Docker services
docker() {
    print_status "Starting Docker services..."
    docker-compose up -d
    print_success "Docker services started!"
    print_status "Services available at:"
    echo "  - Hauptbuch Node: http://localhost:8545"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
}

# Function to stop Docker services
stop() {
    print_status "Stopping Docker services..."
    docker-compose down
    print_success "Docker services stopped!"
}

# Function to show logs
logs() {
    print_status "Showing Docker logs..."
    docker-compose logs -f
}

# Function to open monitoring dashboard
monitor() {
    print_status "Opening monitoring dashboard..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:3000
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:3000
    else
        print_warning "Please open http://localhost:3000 in your browser"
    fi
}

# Main script logic
case "${1:-help}" in
    build)
        build
        ;;
    test)
        test
        ;;
    bench)
        bench
        ;;
    fmt)
        fmt
        ;;
    clippy)
        clippy
        ;;
    audit)
        audit
        ;;
    clean)
        clean
        ;;
    dev)
        dev
        ;;
    docker)
        docker
        ;;
    stop)
        stop
        ;;
    logs)
        logs
        ;;
    monitor)
        monitor
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
