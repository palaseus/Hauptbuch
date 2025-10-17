#!/bin/bash

# Hauptbuch Quick Start Script
# This script provides a quick way to get Hauptbuch running

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
MODE=${1:-development}
SKIP_CHECKS=${2:-false}

# Welcome message
show_welcome() {
    echo "🚀 Welcome to Hauptbuch!"
    echo "========================="
    echo ""
    echo "Hauptbuch is a secure and efficient Proof of Stake blockchain"
    echo "with quantum-resistant cryptography and advanced features."
    echo ""
    echo "This script will help you get started quickly."
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        print_error "Rust is not installed. Please install Rust first:"
        echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
        echo "  Install Docker: https://docs.docker.com/get-docker/"
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install Rust components
    rustup component add rustfmt clippy
    
    # Install Rust tools
    cargo install cargo-watch cargo-audit cargo-deny
    
    print_success "Dependencies installed"
}

# Setup project
setup_project() {
    print_status "Setting up project..."
    
    # Create directories
    mkdir -p data logs monitoring/data/prometheus monitoring/data/grafana
    
    # Copy environment file
    if [ ! -f ".env" ]; then
        cp env.example .env
        print_success "Environment file created: .env"
    fi
    
    # Set permissions
    chmod 777 monitoring/data/prometheus
    chmod 777 monitoring/data/grafana
    
    print_success "Project setup completed"
}

# Build project
build_project() {
    print_status "Building project..."
    
    # Clean build
    cargo clean
    
    # Build project
    cargo build --release
    
    print_success "Project built successfully"
}

# Run tests
run_tests() {
    print_status "Running tests..."
    
    # Run unit tests
    cargo test --lib
    
    # Run integration tests
    cargo test --test integration
    
    print_success "Tests completed"
}

# Start services
start_services() {
    print_status "Starting services..."
    
    # Start Docker services
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Check health
    ./scripts/health-check.sh
    
    print_success "Services started successfully"
}

# Show status
show_status() {
    print_status "Current status:"
    echo ""
    
    # Show Docker status
    echo "Docker containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep hauptbuch || echo "  No Hauptbuch containers running"
    echo ""
    
    # Show ports
    echo "Available services:"
    echo "  - Hauptbuch Node: http://localhost:8545"
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
    echo ""
    
    # Show logs
    echo "Recent logs:"
    docker-compose logs --tail=10
    echo ""
}

# Show next steps
show_next_steps() {
    print_status "Next steps:"
    echo ""
    echo "1. Check the status:"
    echo "   ./scripts/status.sh"
    echo ""
    echo "2. View logs:"
    echo "   ./scripts/dev.sh logs"
    echo ""
    echo "3. Open monitoring dashboard:"
    echo "   ./scripts/dev.sh monitor"
    echo ""
    echo "4. Run benchmarks:"
    echo "   ./scripts/benchmark.sh"
    echo ""
    echo "5. Run tests:"
    echo "   ./scripts/test.sh"
    echo ""
    echo "6. Stop services:"
    echo "   ./scripts/dev.sh stop"
    echo ""
    echo "7. Clean up:"
    echo "   ./scripts/clean.sh"
    echo ""
}

# Development mode
development_mode() {
    print_status "Setting up development mode..."
    
    # Install development dependencies
    cargo install cargo-watch cargo-expand
    
    # Start development server
    print_status "Starting development server..."
    cargo watch -x run
}

# Production mode
production_mode() {
    print_status "Setting up production mode..."
    
    # Build for production
    cargo build --release
    
    # Start production services
    docker-compose -f docker-compose.prod.yml up -d
    
    print_success "Production mode started"
}

# Testing mode
testing_mode() {
    print_status "Setting up testing mode..."
    
    # Run all tests
    ./scripts/test.sh all
    
    # Run benchmarks
    ./scripts/benchmark.sh all
    
    print_success "Testing mode completed"
}

# Main quick start function
main() {
    show_welcome
    
    if [ "$SKIP_CHECKS" != "true" ]; then
        check_prerequisites
    fi
    
    install_dependencies
    setup_project
    build_project
    run_tests
    start_services
    show_status
    show_next_steps
    
    case $MODE in
        "development")
            development_mode
            ;;
        "production")
            production_mode
            ;;
        "testing")
            testing_mode
            ;;
        *)
            print_warning "Unknown mode: $MODE"
            echo "Valid modes: development, production, testing"
            ;;
    esac
    
    print_success "✅ Hauptbuch quick start completed!"
}

# Handle command line arguments
case "${1:-help}" in
    "development"|"production"|"testing")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Quick Start Script"
        echo ""
        echo "Usage: $0 [MODE] [SKIP_CHECKS]"
        echo ""
        echo "Modes:"
        echo "  development  - Development mode (default)"
        echo "  production   - Production mode"
        echo "  testing      - Testing mode"
        echo ""
        echo "Parameters:"
        echo "  SKIP_CHECKS  - Skip prerequisite checks (default: false)"
        echo ""
        echo "Examples:"
        echo "  $0                    # Development mode"
        echo "  $0 production         # Production mode"
        echo "  $0 testing           # Testing mode"
        echo "  $0 development true   # Skip checks"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
