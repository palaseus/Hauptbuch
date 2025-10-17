#!/bin/bash

# Hauptbuch Install Script
# This script installs Hauptbuch on the system

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
INSTALL_TYPE=${1:-all}
INSTALL_DIR=${2:-/opt/hauptbuch}
USER=${3:-hauptbuch}
GROUP=${4:-hauptbuch}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This script only supports Linux"
        exit 1
    fi
    
    # Check architecture
    local arch=$(uname -m)
    if [[ "$arch" != "x86_64" ]]; then
        print_error "This script only supports x86_64 architecture"
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Install system dependencies
install_system_dependencies() {
    print_status "Installing system dependencies..."
    
    # Update package list
    apt-get update
    
    # Install essential packages
    apt-get install -y \
        build-essential \
        pkg-config \
        libssl-dev \
        libclang-dev \
        cmake \
        git \
        curl \
        wget \
        docker.io \
        docker-compose \
        ufw \
        htop \
        vim \
        nano
    
    print_success "System dependencies installed"
}

# Install Rust
install_rust() {
    print_status "Installing Rust..."
    
    # Install Rust
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    
    # Add Rust to PATH
    source ~/.cargo/env
    
    # Install Rust components
    rustup component add rustfmt clippy
    
    print_success "Rust installed"
}

# Create user and group
create_user_group() {
    print_status "Creating user and group..."
    
    # Create group
    if ! getent group $GROUP > /dev/null 2>&1; then
        groupadd $GROUP
        print_success "Group $GROUP created"
    else
        print_warning "Group $GROUP already exists"
    fi
    
    # Create user
    if ! getent passwd $USER > /dev/null 2>&1; then
        useradd -r -s /bin/false -g $GROUP $USER
        print_success "User $USER created"
    else
        print_warning "User $USER already exists"
    fi
}

# Create directories
create_directories() {
    print_status "Creating directories..."
    
    # Create main directory
    mkdir -p $INSTALL_DIR
    
    # Create subdirectories
    mkdir -p $INSTALL_DIR/{bin,lib,etc,var,log,data,backups}
    
    # Set permissions
    chown -R $USER:$GROUP $INSTALL_DIR
    chmod 755 $INSTALL_DIR
    
    print_success "Directories created"
}

# Install Hauptbuch
install_hauptbuch() {
    print_status "Installing Hauptbuch..."
    
    # Build Hauptbuch
    cargo build --release
    
    # Install binary
    cp target/release/hauptbuch $INSTALL_DIR/bin/
    chmod +x $INSTALL_DIR/bin/hauptbuch
    
    # Install configuration
    cp config.toml $INSTALL_DIR/etc/
    cp .env $INSTALL_DIR/etc/ 2>/dev/null || true
    
    # Install scripts
    cp -r scripts/ $INSTALL_DIR/
    chmod +x $INSTALL_DIR/scripts/*.sh
    
    # Install Docker configuration
    cp docker-compose.yml $INSTALL_DIR/
    cp Dockerfile $INSTALL_DIR/
    
    # Install monitoring configuration
    cp -r monitoring/ $INSTALL_DIR/
    
    # Set permissions
    chown -R $USER:$GROUP $INSTALL_DIR
    
    print_success "Hauptbuch installed"
}

# Install systemd service
install_systemd_service() {
    print_status "Installing systemd service..."
    
    # Create systemd service file
    cat > /etc/systemd/system/hauptbuch.service << EOF
[Unit]
Description=Hauptbuch Blockchain Node
After=network.target

[Service]
Type=simple
User=$USER
Group=$GROUP
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/bin/hauptbuch
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable service
    systemctl enable hauptbuch
    
    print_success "Systemd service installed"
}

# Configure firewall
configure_firewall() {
    print_status "Configuring firewall..."
    
    # Enable UFW
    ufw --force enable
    
    # Allow SSH
    ufw allow ssh
    
    # Allow Hauptbuch ports
    ufw allow 30303/tcp
    ufw allow 8545/tcp
    ufw allow 8546/tcp
    
    print_success "Firewall configured"
}

# Install monitoring
install_monitoring() {
    print_status "Installing monitoring..."
    
    # Install Prometheus
    wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz
    tar xzf prometheus-2.45.0.linux-amd64.tar.gz
    mv prometheus-2.45.0.linux-amd64 /opt/prometheus
    
    # Install Grafana
    wget https://dl.grafana.com/oss/release/grafana-10.0.0.linux-amd64.tar.gz
    tar xzf grafana-10.0.0.linux-amd64.tar.gz
    mv grafana-10.0.0 /opt/grafana
    
    # Create systemd services
    cat > /etc/systemd/system/prometheus.service << EOF
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
User=$USER
Group=$GROUP
WorkingDirectory=/opt/prometheus
ExecStart=/opt/prometheus/prometheus
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    cat > /etc/systemd/system/grafana.service << EOF
[Unit]
Description=Grafana
After=network.target

[Service]
Type=simple
User=$USER
Group=$GROUP
WorkingDirectory=/opt/grafana
ExecStart=/opt/grafana/bin/grafana-server
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd
    systemctl daemon-reload
    
    # Enable services
    systemctl enable prometheus
    systemctl enable grafana
    
    print_success "Monitoring installed"
}

# Install all
install_all() {
    print_status "Installing everything..."
    
    install_system_dependencies
    install_rust
    create_user_group
    create_directories
    install_hauptbuch
    install_systemd_service
    configure_firewall
    install_monitoring
    
    print_success "Everything installed successfully"
}

# Show installation status
show_status() {
    print_status "Installation status:"
    echo "  Type: $INSTALL_TYPE"
    echo "  Directory: $INSTALL_DIR"
    echo "  User: $USER"
    echo "  Group: $GROUP"
    echo ""
    
    echo "Services:"
    echo "  Hauptbuch: $(systemctl is-enabled hauptbuch 2>/dev/null || echo "disabled")"
    echo "  Prometheus: $(systemctl is-enabled prometheus 2>/dev/null || echo "disabled")"
    echo "  Grafana: $(systemctl is-enabled grafana 2>/dev/null || echo "disabled")"
    echo ""
    
    echo "Directories:"
    echo "  Install: $INSTALL_DIR"
    echo "  Data: $INSTALL_DIR/data"
    echo "  Logs: $INSTALL_DIR/log"
    echo "  Backups: $INSTALL_DIR/backups"
    echo ""
    
    echo "Ports:"
    echo "  P2P: 30303"
    echo "  RPC: 8545"
    echo "  WebSocket: 8546"
    echo "  Prometheus: 9090"
    echo "  Grafana: 3000"
}

# Main installation function
main() {
    print_status "🚀 Starting Hauptbuch installation..."
    
    check_root
    check_requirements
    
    case $INSTALL_TYPE in
        "dependencies")
            install_system_dependencies
            ;;
        "rust")
            install_rust
            ;;
        "user")
            create_user_group
            ;;
        "directories")
            create_directories
            ;;
        "hauptbuch")
            install_hauptbuch
            ;;
        "service")
            install_systemd_service
            ;;
        "firewall")
            configure_firewall
            ;;
        "monitoring")
            install_monitoring
            ;;
        "all")
            install_all
            ;;
        *)
            print_error "Unknown install type: $INSTALL_TYPE"
            echo "Valid types: dependencies, rust, user, directories, hauptbuch, service, firewall, monitoring, all"
            exit 1
            ;;
    esac
    
    show_status
    print_success "✅ Installation completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "dependencies"|"rust"|"user"|"directories"|"hauptbuch"|"service"|"firewall"|"monitoring"|"all")
        main "$@"
        ;;
    "status")
        show_status
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Install Script"
        echo ""
        echo "Usage: $0 [INSTALL_TYPE] [INSTALL_DIR] [USER] [GROUP]"
        echo ""
        echo "Install Types:"
        echo "  dependencies - Install system dependencies"
        echo "  rust         - Install Rust toolchain"
        echo "  user         - Create user and group"
        echo "  directories  - Create directories"
        echo "  hauptbuch    - Install Hauptbuch"
        echo "  service      - Install systemd service"
        echo "  firewall     - Configure firewall"
        echo "  monitoring   - Install monitoring"
        echo "  all          - Install everything (default)"
        echo ""
        echo "Commands:"
        echo "  status       - Show installation status"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  INSTALL_DIR  - Installation directory (default: /opt/hauptbuch)"
        echo "  USER         - User name (default: hauptbuch)"
        echo "  GROUP        - Group name (default: hauptbuch)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
