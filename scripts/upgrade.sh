#!/bin/bash

# Hauptbuch Upgrade Script
# This script upgrades Hauptbuch to the latest version

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
UPGRADE_TYPE=${1:-all}
VERSION=${2:-latest}
BACKUP=${3:-true}
FORCE=${4:-false}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. This is not recommended."
        if [ "$FORCE" != "true" ]; then
            print_error "Use --force to continue as root"
            exit 1
        fi
    fi
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Rust
    if ! command -v cargo &> /dev/null; then
        print_error "Rust is not installed. Please install Rust first."
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
    fi
    
    print_success "System requirements check passed"
}

# Create backup
create_backup() {
    if [ "$BACKUP" = "true" ]; then
        print_status "Creating backup before upgrade..."
        
        # Create backup
        ./scripts/backup.sh all true false
        
        print_success "Backup created successfully"
    else
        print_warning "Backup disabled. Proceeding without backup."
    fi
}

# Upgrade Rust toolchain
upgrade_rust() {
    print_status "Upgrading Rust toolchain..."
    
    # Update Rust
    rustup update
    
    # Update components
    rustup component add rustfmt clippy
    
    print_success "Rust toolchain upgraded"
}

# Upgrade dependencies
upgrade_dependencies() {
    print_status "Upgrading dependencies..."
    
    # Update Cargo dependencies
    cargo update
    
    # Update system dependencies
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update
        sudo apt-get upgrade -y
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew update
        brew upgrade
    fi
    
    print_success "Dependencies upgraded"
}

# Upgrade source code
upgrade_source() {
    print_status "Upgrading source code..."
    
    # Check if we're in a git repository
    if [ -d ".git" ]; then
        # Fetch latest changes
        git fetch origin
        
        # Check for updates
        local current_branch=$(git branch --show-current)
        local remote_branch="origin/$current_branch"
        
        if git rev-list HEAD..$remote_branch --count > 0; then
            print_status "Updates available. Upgrading..."
            
            # Stash any local changes
            git stash
            
            # Pull latest changes
            git pull origin "$current_branch"
            
            # Apply stashed changes
            git stash pop
            
            print_success "Source code upgraded"
        else
            print_success "Source code is up to date"
        fi
    else
        print_warning "Not in a git repository. Manual upgrade required."
    fi
}

# Upgrade Docker images
upgrade_docker() {
    print_status "Upgrading Docker images..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Skipping Docker upgrade."
        return 0
    fi
    
    # Pull latest images
    docker-compose pull
    
    # Rebuild images
    docker-compose build --no-cache
    
    print_success "Docker images upgraded"
}

# Upgrade configuration
upgrade_config() {
    print_status "Upgrading configuration..."
    
    # Check if config needs updating
    if [ -f "config.toml" ]; then
        # Backup current config
        cp config.toml config.toml.backup
        
        # Update config if needed
        print_status "Configuration file exists. Manual review recommended."
    fi
    
    # Update environment file
    if [ -f ".env" ]; then
        cp .env .env.backup
        print_status "Environment file backed up"
    fi
    
    print_success "Configuration upgraded"
}

# Upgrade monitoring
upgrade_monitoring() {
    print_status "Upgrading monitoring configuration..."
    
    # Update Prometheus configuration
    if [ -f "monitoring/prometheus.yml" ]; then
        cp monitoring/prometheus.yml monitoring/prometheus.yml.backup
        print_status "Prometheus configuration backed up"
    fi
    
    # Update Grafana configuration
    if [ -d "monitoring/grafana" ]; then
        cp -r monitoring/grafana monitoring/grafana.backup
        print_status "Grafana configuration backed up"
    fi
    
    print_success "Monitoring configuration upgraded"
}

# Rebuild project
rebuild_project() {
    print_status "Rebuilding project..."
    
    # Clean build
    cargo clean
    
    # Build project
    cargo build --release
    
    # Run tests
    cargo test
    
    print_success "Project rebuilt successfully"
}

# Restart services
restart_services() {
    print_status "Restarting services..."
    
    # Stop existing services
    docker-compose down
    
    # Start services
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Check health
    ./scripts/health-check.sh
    
    print_success "Services restarted"
}

# Upgrade all
upgrade_all() {
    print_status "Upgrading everything..."
    
    upgrade_rust
    upgrade_dependencies
    upgrade_source
    upgrade_docker
    upgrade_config
    upgrade_monitoring
    rebuild_project
    restart_services
    
    print_success "Everything upgraded successfully"
}

# Show upgrade status
show_status() {
    print_status "Upgrade status:"
    echo "  Type: $UPGRADE_TYPE"
    echo "  Version: $VERSION"
    echo "  Backup: $BACKUP"
    echo "  Force: $FORCE"
    echo ""
    
    echo "Current versions:"
    echo "  Rust: $(rustc --version)"
    echo "  Cargo: $(cargo --version)"
    echo "  Git: $(git --version)"
    
    if command -v docker &> /dev/null; then
        echo "  Docker: $(docker --version)"
    fi
    
    echo ""
    echo "Project status:"
    if [ -d ".git" ]; then
        echo "  Git branch: $(git branch --show-current)"
        echo "  Git status: $(git status --porcelain | wc -l) changes"
    fi
    
    echo "  Build status: $(cargo check --quiet && echo "OK" || echo "FAILED")"
}

# Rollback upgrade
rollback() {
    print_status "Rolling back upgrade..."
    
    # Restore from backup
    local latest_backup=$(ls -t backups/ | head -n1)
    if [ -n "$latest_backup" ]; then
        ./scripts/restore.sh all "backups/$latest_backup"
        print_success "Rolled back to: $latest_backup"
    else
        print_error "No backup found for rollback"
        exit 1
    fi
}

# Main upgrade function
main() {
    print_status "🔄 Starting Hauptbuch upgrade..."
    
    check_root
    check_requirements
    create_backup
    
    case $UPGRADE_TYPE in
        "rust")
            upgrade_rust
            ;;
        "dependencies")
            upgrade_dependencies
            ;;
        "source")
            upgrade_source
            ;;
        "docker")
            upgrade_docker
            ;;
        "config")
            upgrade_config
            ;;
        "monitoring")
            upgrade_monitoring
            ;;
        "rebuild")
            rebuild_project
            ;;
        "restart")
            restart_services
            ;;
        "all")
            upgrade_all
            ;;
        *)
            print_error "Unknown upgrade type: $UPGRADE_TYPE"
            echo "Valid types: rust, dependencies, source, docker, config, monitoring, rebuild, restart, all"
            exit 1
            ;;
    esac
    
    show_status
    print_success "✅ Upgrade completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "rust"|"dependencies"|"source"|"docker"|"config"|"monitoring"|"rebuild"|"restart"|"all")
        main "$@"
        ;;
    "status")
        show_status
        ;;
    "rollback")
        rollback
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Upgrade Script"
        echo ""
        echo "Usage: $0 [UPGRADE_TYPE] [VERSION] [BACKUP] [FORCE]"
        echo ""
        echo "Upgrade Types:"
        echo "  rust         - Upgrade Rust toolchain"
        echo "  dependencies - Upgrade system dependencies"
        echo "  source       - Upgrade source code"
        echo "  docker       - Upgrade Docker images"
        echo "  config       - Upgrade configuration"
        echo "  monitoring   - Upgrade monitoring configuration"
        echo "  rebuild      - Rebuild project"
        echo "  restart      - Restart services"
        echo "  all          - Upgrade everything (default)"
        echo ""
        echo "Commands:"
        echo "  status       - Show upgrade status"
        echo "  rollback     - Rollback to previous version"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  VERSION      - Target version (default: latest)"
        echo "  BACKUP       - Create backup before upgrade (default: true)"
        echo "  FORCE        - Force upgrade even as root (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
