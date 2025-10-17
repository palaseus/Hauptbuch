#!/bin/bash

# Hauptbuch Dependencies Installation Script
# This script installs all necessary dependencies for Hauptbuch

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

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    print_status "Detected OS: $OS"
}

# Install Rust
install_rust() {
    if command -v cargo &> /dev/null; then
        print_success "Rust is already installed"
        return
    fi
    
    print_status "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
    print_success "Rust installed successfully"
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    case $OS in
        "linux")
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                pkg-config \
                libssl-dev \
                libclang-dev \
                cmake \
                git \
                curl \
                wget \
                docker.io \
                docker-compose
            ;;
        "macos")
            if ! command -v brew &> /dev/null; then
                print_error "Homebrew is not installed. Please install Homebrew first."
                exit 1
            fi
            brew install pkg-config openssl cmake git curl wget docker docker-compose
            ;;
        "windows")
            print_warning "Windows support is limited. Please install dependencies manually."
            ;;
        *)
            print_error "Unsupported OS: $OS"
            exit 1
            ;;
    esac
    
    print_success "System dependencies installed"
}

# Install Rust components
install_rust_components() {
    print_status "Installing Rust components..."
    
    rustup component add rustfmt clippy
    rustup toolchain install nightly
    
    print_success "Rust components installed"
}

# Install Rust tools
install_rust_tools() {
    print_status "Installing Rust tools..."
    
    cargo install cargo-watch cargo-expand cargo-audit cargo-deny cargo-tarpaulin
    
    print_success "Rust tools installed"
}

# Install Docker
install_docker() {
    if command -v docker &> /dev/null; then
        print_success "Docker is already installed"
        return
    fi
    
    print_status "Installing Docker..."
    
    case $OS in
        "linux")
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            ;;
        "macos")
            print_warning "Please install Docker Desktop for Mac from https://www.docker.com/products/docker-desktop"
            ;;
        "windows")
            print_warning "Please install Docker Desktop for Windows from https://www.docker.com/products/docker-desktop"
            ;;
    esac
    
    print_success "Docker installed"
}

# Install monitoring tools
install_monitoring() {
    print_status "Installing monitoring tools..."
    
    case $OS in
        "linux")
            # Install Prometheus
            wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
            tar xzf prometheus-2.45.0.linux-amd64.tar.gz
            sudo mv prometheus-2.45.0.linux-amd64 /opt/prometheus
            
            # Install Grafana
            wget https://dl.grafana.com/oss/release/grafana-10.0.0.linux-amd64.tar.gz
            tar xzf grafana-10.0.0.linux-amd64.tar.gz
            sudo mv grafana-10.0.0 /opt/grafana
            ;;
        "macos")
            brew install prometheus grafana
            ;;
        "windows")
            print_warning "Please install monitoring tools manually on Windows"
            ;;
    esac
    
    print_success "Monitoring tools installed"
}

# Main installation function
main() {
    print_status "🚀 Starting Hauptbuch dependencies installation..."
    
    detect_os
    install_rust
    install_system_deps
    install_rust_components
    install_rust_tools
    install_docker
    install_monitoring
    
    print_success "✅ All dependencies installed successfully!"
    print_status "Next steps:"
    echo "1. Run './scripts/setup.sh' to set up the project"
    echo "2. Run 'make dev' to start development"
    echo "3. Run 'docker-compose up' to start the full stack"
}

# Run main function
main "$@"
