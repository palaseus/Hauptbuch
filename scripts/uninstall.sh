#!/bin/bash

# Hauptbuch Uninstall Script
# This script uninstalls Hauptbuch from the system

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
UNINSTALL_TYPE=${1:-all}
INSTALL_DIR=${2:-/opt/hauptbuch}
USER=${3:-hauptbuch}
GROUP=${4:-hauptbuch}
KEEP_DATA=${5:-false}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run as root"
        exit 1
    fi
}

# Stop services
stop_services() {
    print_status "Stopping services..."
    
    # Stop Hauptbuch service
    systemctl stop hauptbuch 2>/dev/null || true
    
    # Stop monitoring services
    systemctl stop prometheus 2>/dev/null || true
    systemctl stop grafana 2>/dev/null || true
    
    # Stop Docker services
    docker-compose down 2>/dev/null || true
    
    print_success "Services stopped"
}

# Remove systemd services
remove_systemd_services() {
    print_status "Removing systemd services..."
    
    # Disable services
    systemctl disable hauptbuch 2>/dev/null || true
    systemctl disable prometheus 2>/dev/null || true
    systemctl disable grafana 2>/dev/null || true
    
    # Remove service files
    rm -f /etc/systemd/system/hauptbuch.service
    rm -f /etc/systemd/system/prometheus.service
    rm -f /etc/systemd/system/grafana.service
    
    # Reload systemd
    systemctl daemon-reload
    
    print_success "Systemd services removed"
}

# Remove Hauptbuch
remove_hauptbuch() {
    print_status "Removing Hauptbuch..."
    
    # Remove binary
    rm -f $INSTALL_DIR/bin/hauptbuch
    
    # Remove configuration
    rm -f $INSTALL_DIR/etc/config.toml
    rm -f $INSTALL_DIR/etc/.env
    
    # Remove scripts
    rm -rf $INSTALL_DIR/scripts
    
    # Remove Docker configuration
    rm -f $INSTALL_DIR/docker-compose.yml
    rm -f $INSTALL_DIR/Dockerfile
    
    # Remove monitoring configuration
    rm -rf $INSTALL_DIR/monitoring
    
    print_success "Hauptbuch removed"
}

# Remove monitoring
remove_monitoring() {
    print_status "Removing monitoring..."
    
    # Remove Prometheus
    rm -rf /opt/prometheus
    
    # Remove Grafana
    rm -rf /opt/grafana
    
    # Remove monitoring data
    if [ "$KEEP_DATA" != "true" ]; then
        rm -rf $INSTALL_DIR/monitoring
    fi
    
    print_success "Monitoring removed"
}

# Remove data
remove_data() {
    if [ "$KEEP_DATA" != "true" ]; then
        print_status "Removing data..."
        
        # Remove data directory
        rm -rf $INSTALL_DIR/data
        
        # Remove logs
        rm -rf $INSTALL_DIR/log
        
        # Remove backups
        rm -rf $INSTALL_DIR/backups
        
        print_success "Data removed"
    else
        print_warning "Keeping data as requested"
    fi
}

# Remove directories
remove_directories() {
    print_status "Removing directories..."
    
    # Remove main directory
    if [ -d "$INSTALL_DIR" ]; then
        rm -rf $INSTALL_DIR
        print_success "Installation directory removed"
    else
        print_warning "Installation directory not found"
    fi
}

# Remove user and group
remove_user_group() {
    print_status "Removing user and group..."
    
    # Remove user
    if getent passwd $USER > /dev/null 2>&1; then
        userdel $USER
        print_success "User $USER removed"
    else
        print_warning "User $USER not found"
    fi
    
    # Remove group
    if getent group $GROUP > /dev/null 2>&1; then
        groupdel $GROUP
        print_success "Group $GROUP removed"
    else
        print_warning "Group $GROUP not found"
    fi
}

# Remove Docker volumes
remove_docker_volumes() {
    print_status "Removing Docker volumes..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    # List Hauptbuch volumes
    local volumes=$(docker volume ls --filter "name=hauptbuch" --format "{{.Name}}")
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            docker volume rm $volume 2>/dev/null || true
        done
        print_success "Docker volumes removed"
    else
        print_warning "No Hauptbuch volumes found"
    fi
}

# Remove Docker images
remove_docker_images() {
    print_status "Removing Docker images..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    # Remove Hauptbuch images
    docker images --filter "reference=hauptbuch*" --format "{{.ID}}" | xargs -r docker rmi -f 2>/dev/null || true
    
    print_success "Docker images removed"
}

# Remove system dependencies
remove_system_dependencies() {
    print_status "Removing system dependencies..."
    
    # Remove packages
    apt-get remove -y \
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
    
    print_success "System dependencies removed"
}

# Remove Rust
remove_rust() {
    print_status "Removing Rust..."
    
    # Remove Rust
    rm -rf ~/.rustup
    rm -rf ~/.cargo
    
    print_success "Rust removed"
}

# Remove all
remove_all() {
    print_status "Removing everything..."
    
    stop_services
    remove_systemd_services
    remove_hauptbuch
    remove_monitoring
    remove_data
    remove_directories
    remove_user_group
    remove_docker_volumes
    remove_docker_images
    
    print_success "Everything removed"
}

# Show uninstallation status
show_status() {
    print_status "Uninstallation status:"
    echo "  Type: $UNINSTALL_TYPE"
    echo "  Directory: $INSTALL_DIR"
    echo "  User: $USER"
    echo "  Group: $GROUP"
    echo "  Keep Data: $KEEP_DATA"
    echo ""
    
    echo "Services:"
    echo "  Hauptbuch: $(systemctl is-enabled hauptbuch 2>/dev/null || echo "disabled")"
    echo "  Prometheus: $(systemctl is-enabled prometheus 2>/dev/null || echo "disabled")"
    echo "  Grafana: $(systemctl is-enabled grafana 2>/dev/null || echo "disabled")"
    echo ""
    
    echo "Directories:"
    echo "  Install: $([ -d "$INSTALL_DIR" ] && echo "exists" || echo "removed")"
    echo "  Data: $([ -d "$INSTALL_DIR/data" ] && echo "exists" || echo "removed")"
    echo "  Logs: $([ -d "$INSTALL_DIR/log" ] && echo "exists" || echo "removed")"
    echo "  Backups: $([ -d "$INSTALL_DIR/backups" ] && echo "exists" || echo "removed")"
    echo ""
    
    echo "User/Group:"
    echo "  User: $(getent passwd $USER > /dev/null 2>&1 && echo "exists" || echo "removed")"
    echo "  Group: $(getent group $GROUP > /dev/null 2>&1 && echo "exists" || echo "removed")"
}

# Main uninstallation function
main() {
    print_status "🗑️ Starting Hauptbuch uninstallation..."
    
    check_root
    
    case $UNINSTALL_TYPE in
        "services")
            stop_services
            remove_systemd_services
            ;;
        "hauptbuch")
            remove_hauptbuch
            ;;
        "monitoring")
            remove_monitoring
            ;;
        "data")
            remove_data
            ;;
        "directories")
            remove_directories
            ;;
        "user")
            remove_user_group
            ;;
        "docker")
            remove_docker_volumes
            remove_docker_images
            ;;
        "dependencies")
            remove_system_dependencies
            ;;
        "rust")
            remove_rust
            ;;
        "all")
            remove_all
            ;;
        *)
            print_error "Unknown uninstall type: $UNINSTALL_TYPE"
            echo "Valid types: services, hauptbuch, monitoring, data, directories, user, docker, dependencies, rust, all"
            exit 1
            ;;
    esac
    
    show_status
    print_success "✅ Uninstallation completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "services"|"hauptbuch"|"monitoring"|"data"|"directories"|"user"|"docker"|"dependencies"|"rust"|"all")
        main "$@"
        ;;
    "status")
        show_status
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Uninstall Script"
        echo ""
        echo "Usage: $0 [UNINSTALL_TYPE] [INSTALL_DIR] [USER] [GROUP] [KEEP_DATA]"
        echo ""
        echo "Uninstall Types:"
        echo "  services     - Remove systemd services"
        echo "  hauptbuch    - Remove Hauptbuch"
        echo "  monitoring   - Remove monitoring"
        echo "  data         - Remove data"
        echo "  directories  - Remove directories"
        echo "  user         - Remove user and group"
        echo "  docker       - Remove Docker volumes and images"
        echo "  dependencies - Remove system dependencies"
        echo "  rust         - Remove Rust"
        echo "  all          - Remove everything (default)"
        echo ""
        echo "Commands:"
        echo "  status       - Show uninstallation status"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  INSTALL_DIR  - Installation directory (default: /opt/hauptbuch)"
        echo "  USER         - User name (default: hauptbuch)"
        echo "  GROUP        - Group name (default: hauptbuch)"
        echo "  KEEP_DATA    - Keep data files (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
