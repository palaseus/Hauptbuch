#!/bin/bash

# Hauptbuch Update Script
# This script updates Hauptbuch to the latest version

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
UPDATE_TYPE=${1:-all}
BACKUP=${2:-true}
FORCE=${3:-false}

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
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Some features may not work."
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    print_success "System requirements check passed"
}

# Backup current installation
backup_current() {
    if [ "$BACKUP" = "true" ]; then
        print_status "Creating backup of current installation..."
        
        # Create backup
        ./scripts/backup.sh all true false
        
        print_success "Backup created successfully"
    else
        print_warning "Backup disabled. Proceeding without backup."
    fi
}

# Update Rust toolchain
update_rust() {
    print_status "Updating Rust toolchain..."
    
    # Update Rust
    rustup update
    
    # Update components
    rustup component add rustfmt clippy
    
    print_success "Rust toolchain updated"
}

# Update dependencies
update_dependencies() {
    print_status "Updating dependencies..."
    
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
    
    print_success "Dependencies updated"
}

# Update source code
update_source() {
    print_status "Updating source code..."
    
    # Check if we're in a git repository
    if [ -d ".git" ]; then
        # Fetch latest changes
        git fetch origin
        
        # Check for updates
        local current_branch=$(git branch --show-current)
        local remote_branch="origin/$current_branch"
        
        if git rev-list HEAD..$remote_branch --count > 0; then
            print_status "Updates available. Updating..."
            
            # Stash any local changes
            git stash
            
            # Pull latest changes
            git pull origin "$current_branch"
            
            # Apply stashed changes
            git stash pop
            
            print_success "Source code updated"
        else
            print_success "Source code is up to date"
        fi
    else
        print_warning "Not in a git repository. Manual update required."
    fi
}

# Update Docker images
update_docker() {
    print_status "Updating Docker images..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed. Skipping Docker update."
        return 0
    fi
    
    # Pull latest images
    docker-compose pull
    
    # Rebuild images
    docker-compose build --no-cache
    
    print_success "Docker images updated"
}

# Update configuration
update_config() {
    print_status "Updating configuration..."
    
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
    
    print_success "Configuration updated"
}

# Update monitoring
update_monitoring() {
    print_status "Updating monitoring configuration..."
    
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
    
    print_success "Monitoring configuration updated"
}

# Rebuild project
rebuild_project() {
    print_status "Rebuilding project..."
    
    # Clean build artifacts
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

# Update all
update_all() {
    print_status "Updating everything..."
    
    update_rust
    update_dependencies
    update_source
    update_docker
    update_config
    update_monitoring
    rebuild_project
    restart_services
    
    print_success "Everything updated successfully"
}

# Show update status
show_status() {
    print_status "Update status:"
    echo "  Type: $UPDATE_TYPE"
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

# Rollback update
rollback() {
    print_status "Rolling back update..."
    
    # Restore from backup
    local latest_backup=$(ls -t backups/ | head -n1)
    if [ -n "$latest_backup" ]; then
        ./scripts/backup.sh restore "backups/$latest_backup"
        print_success "Rolled back to: $latest_backup"
    else
        print_error "No backup found for rollback"
        exit 1
    fi
}

# Main update function
main() {
    print_status "🔄 Starting Hauptbuch update..."
    
    check_root
    check_requirements
    backup_current
    
    case $UPDATE_TYPE in
        "rust")
            update_rust
            ;;
        "dependencies")
            update_dependencies
            ;;
        "source")
            update_source
            ;;
        "docker")
            update_docker
            ;;
        "config")
            update_config
            ;;
        "monitoring")
            update_monitoring
            ;;
        "rebuild")
            rebuild_project
            ;;
        "restart")
            restart_services
            ;;
        "all")
            update_all
            ;;
        *)
            print_error "Unknown update type: $UPDATE_TYPE"
            echo "Valid types: rust, dependencies, source, docker, config, monitoring, rebuild, restart, all"
            exit 1
            ;;
    esac
    
    show_status
    print_success "✅ Update completed successfully!"
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
        echo "Hauptbuch Update Script"
        echo ""
        echo "Usage: $0 [UPDATE_TYPE] [BACKUP] [FORCE]"
        echo ""
        echo "Update Types:"
        echo "  rust         - Update Rust toolchain"
        echo "  dependencies - Update system dependencies"
        echo "  source       - Update source code"
        echo "  docker       - Update Docker images"
        echo "  config       - Update configuration"
        echo "  monitoring   - Update monitoring configuration"
        echo "  rebuild      - Rebuild project"
        echo "  restart      - Restart services"
        echo "  all          - Update everything (default)"
        echo ""
        echo "Commands:"
        echo "  status       - Show update status"
        echo "  rollback     - Rollback to previous version"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  BACKUP       - Create backup before update (default: true)"
        echo "  FORCE        - Force update even as root (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
