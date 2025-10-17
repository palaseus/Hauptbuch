#!/bin/bash

# Hauptbuch Rollback Script
# This script rolls back Hauptbuch to a previous version

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
ROLLBACK_TYPE=${1:-all}
BACKUP_PATH=${2:-""}
FORCE=${3:-false}

# Validate backup path
validate_backup_path() {
    if [ -z "$BACKUP_PATH" ]; then
        # Find latest backup
        local latest_backup=$(ls -t backups/ 2>/dev/null | head -n1)
        if [ -n "$latest_backup" ]; then
            BACKUP_PATH="backups/$latest_backup"
            print_status "Using latest backup: $BACKUP_PATH"
        else
            print_error "No backup path specified and no backups found"
            exit 1
        fi
    fi
    
    if [ ! -e "$BACKUP_PATH" ]; then
        print_error "Backup not found: $BACKUP_PATH"
        exit 1
    fi
    
    print_success "Backup path validated: $BACKUP_PATH"
}

# Rollback data
rollback_data() {
    print_status "Rolling back data..."
    
    # Check if data directory exists
    if [ -d "data" ] && [ "$FORCE" != "true" ]; then
        print_warning "Data directory already exists. Use --force to overwrite."
        return 0
    fi
    
    # Remove existing data directory
    if [ -d "data" ]; then
        rm -rf data
    fi
    
    # Restore from backup
    if [ -f "$BACKUP_PATH/data.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/data.tar.gz"
        print_success "Data rolled back from: $BACKUP_PATH/data.tar.gz"
    elif [ -d "$BACKUP_PATH/data" ]; then
        cp -r "$BACKUP_PATH/data" .
        print_success "Data rolled back from: $BACKUP_PATH/data/"
    else
        print_warning "No data backup found in: $BACKUP_PATH"
    fi
}

# Rollback configuration
rollback_config() {
    print_status "Rolling back configuration..."
    
    # Rollback config files
    if [ -f "$BACKUP_PATH/config.toml" ]; then
        cp "$BACKUP_PATH/config.toml" .
        print_success "config.toml rolled back"
    fi
    
    if [ -f "$BACKUP_PATH/.env" ]; then
        cp "$BACKUP_PATH/.env" .
        print_success ".env rolled back"
    fi
    
    if [ -f "$BACKUP_PATH/env.example" ]; then
        cp "$BACKUP_PATH/env.example" .
        print_success "env.example rolled back"
    fi
    
    # Rollback Docker configuration
    if [ -f "$BACKUP_PATH/docker-compose.yml" ]; then
        cp "$BACKUP_PATH/docker-compose.yml" .
        print_success "docker-compose.yml rolled back"
    fi
    
    if [ -f "$BACKUP_PATH/Dockerfile" ]; then
        cp "$BACKUP_PATH/Dockerfile" .
        print_success "Dockerfile rolled back"
    fi
    
    # Rollback monitoring configuration
    if [ -d "$BACKUP_PATH/monitoring" ]; then
        cp -r "$BACKUP_PATH/monitoring" .
        print_success "Monitoring configuration rolled back"
    fi
}

# Rollback logs
rollback_logs() {
    print_status "Rolling back logs..."
    
    # Check if logs directory exists
    if [ -d "logs" ] && [ "$FORCE" != "true" ]; then
        print_warning "Logs directory already exists. Use --force to overwrite."
        return 0
    fi
    
    # Remove existing logs directory
    if [ -d "logs" ]; then
        rm -rf logs
    fi
    
    # Restore from backup
    if [ -f "$BACKUP_PATH/logs.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/logs.tar.gz"
        print_success "Logs rolled back from: $BACKUP_PATH/logs.tar.gz"
    elif [ -d "$BACKUP_PATH/logs" ]; then
        cp -r "$BACKUP_PATH/logs" .
        print_success "Logs rolled back from: $BACKUP_PATH/logs/"
    else
        print_warning "No logs backup found in: $BACKUP_PATH"
    fi
}

# Rollback Docker volumes
rollback_docker_volumes() {
    print_status "Rolling back Docker volumes..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    # List Hauptbuch volumes
    local volumes=$(docker volume ls --filter "name=hauptbuch" --format "{{.Name}}")
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            if [ -f "$BACKUP_PATH/$volume.tar.gz" ]; then
                print_status "Rolling back volume: $volume"
                docker run --rm -v "$volume":/data -v "$(pwd)/$BACKUP_PATH":/backup alpine tar xzf "/backup/$volume.tar.gz" -C /data
                print_success "Volume $volume rolled back"
            else
                print_warning "No backup found for volume: $volume"
            fi
        done
    else
        print_warning "No Hauptbuch volumes found"
    fi
}

# Rollback source code
rollback_source() {
    print_status "Rolling back source code..."
    
    # Check if source directory exists
    if [ -d "src" ] && [ "$FORCE" != "true" ]; then
        print_warning "Source directory already exists. Use --force to overwrite."
        return 0
    fi
    
    # Restore from backup
    if [ -f "$BACKUP_PATH/source.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/source.tar.gz"
        print_success "Source code rolled back from: $BACKUP_PATH/source.tar.gz"
    elif [ -d "$BACKUP_PATH/source" ]; then
        cp -r "$BACKUP_PATH/source"/*" .
        print_success "Source code rolled back from: $BACKUP_PATH/source/"
    else
        print_warning "No source backup found in: $BACKUP_PATH"
    fi
}

# Rollback database
rollback_database() {
    print_status "Rolling back database..."
    
    # Check if database directory exists
    if [ -d "data" ] && [ "$FORCE" != "true" ]; then
        print_warning "Database directory already exists. Use --force to overwrite."
        return 0
    fi
    
    # Remove existing database directory
    if [ -d "data" ]; then
        rm -rf data
    fi
    
    # Restore from backup
    if [ -f "$BACKUP_PATH/database.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/database.tar.gz"
        print_success "Database rolled back from: $BACKUP_PATH/database.tar.gz"
    elif [ -d "$BACKUP_PATH/database" ]; then
        cp -r "$BACKUP_PATH/database" data
        print_success "Database rolled back from: $BACKUP_PATH/database/"
    else
        print_warning "No database backup found in: $BACKUP_PATH"
    fi
}

# Rollback monitoring data
rollback_monitoring() {
    print_status "Rolling back monitoring data..."
    
    # Create monitoring directory
    mkdir -p monitoring/data
    
    # Rollback Prometheus data
    if [ -f "$BACKUP_PATH/prometheus-data.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/prometheus-data.tar.gz" -C monitoring/data/
        print_success "Prometheus data rolled back from: $BACKUP_PATH/prometheus-data.tar.gz"
    elif [ -d "$BACKUP_PATH/prometheus-data" ]; then
        cp -r "$BACKUP_PATH/prometheus-data" monitoring/data/
        print_success "Prometheus data rolled back from: $BACKUP_PATH/prometheus-data/"
    else
        print_warning "No Prometheus data backup found in: $BACKUP_PATH"
    fi
    
    # Rollback Grafana data
    if [ -f "$BACKUP_PATH/grafana-data.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/grafana-data.tar.gz" -C monitoring/data/
        print_success "Grafana data rolled back from: $BACKUP_PATH/grafana-data.tar.gz"
    elif [ -d "$BACKUP_PATH/grafana-data" ]; then
        cp -r "$BACKUP_PATH/grafana-data" monitoring/data/
        print_success "Grafana data rolled back from: $BACKUP_PATH/grafana-data/"
    else
        print_warning "No Grafana data backup found in: $BACKUP_PATH"
    fi
}

# Rollback all
rollback_all() {
    print_status "Rolling back complete backup..."
    
    # Rollback all components
    rollback_data
    rollback_config
    rollback_logs
    rollback_docker_volumes
    rollback_source
    rollback_database
    rollback_monitoring
    
    print_success "Complete backup rolled back from: $BACKUP_PATH"
}

# Verify rollback
verify_rollback() {
    print_status "Verifying rollback..."
    
    # Check rolled back components
    local components=("data" "config.toml" "docker-compose.yml" "src")
    local rolled_back_count=0
    
    for component in "${components[@]}"; do
        if [ -e "$component" ]; then
            print_success "Component rolled back: $component"
            rolled_back_count=$((rolled_back_count + 1))
        else
            print_warning "Component not found: $component"
        fi
    done
    
    print_status "Rolled back components: $rolled_back_count/${#components[@]}"
    
    # Check file permissions
    if [ -f "config.toml" ]; then
        local perms=$(stat -c %a config.toml 2>/dev/null || stat -f %A config.toml 2>/dev/null || echo "000")
        if [ "$perms" -le 644 ]; then
            print_success "Configuration file permissions are secure"
        else
            print_warning "Configuration file permissions are too open: $perms"
        fi
    fi
    
    print_success "Rollback verification completed"
}

# Show rollback information
show_rollback_info() {
    print_status "Rollback information:"
    echo "  Backup path: $BACKUP_PATH"
    echo "  Type: $ROLLBACK_TYPE"
    echo "  Force: $FORCE"
    echo ""
    
    if [ -d "$BACKUP_PATH" ]; then
        echo "Backup contents:"
        ls -la "$BACKUP_PATH"
        echo ""
        echo "Backup size: $(du -sh "$BACKUP_PATH" | cut -f1)"
    fi
}

# Main rollback function
main() {
    print_status "🔄 Starting Hauptbuch rollback..."
    
    validate_backup_path
    show_rollback_info
    
    case $ROLLBACK_TYPE in
        "data")
            rollback_data
            ;;
        "config")
            rollback_config
            ;;
        "logs")
            rollback_logs
            ;;
        "docker")
            rollback_docker_volumes
            ;;
        "source")
            rollback_source
            ;;
        "database")
            rollback_database
            ;;
        "monitoring")
            rollback_monitoring
            ;;
        "all")
            rollback_all
            ;;
        *)
            print_error "Unknown rollback type: $ROLLBACK_TYPE"
            echo "Valid types: data, config, logs, docker, source, database, monitoring, all"
            exit 1
            ;;
    esac
    
    verify_rollback
    print_success "✅ Rollback completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "data"|"config"|"logs"|"docker"|"source"|"database"|"monitoring"|"all")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Rollback Script"
        echo ""
        echo "Usage: $0 [ROLLBACK_TYPE] [BACKUP_PATH] [FORCE]"
        echo ""
        echo "Rollback Types:"
        echo "  data        - Rollback data directory"
        echo "  config      - Rollback configuration files"
        echo "  logs        - Rollback log files"
        echo "  docker      - Rollback Docker volumes"
        echo "  source      - Rollback source code"
        echo "  database    - Rollback database"
        echo "  monitoring  - Rollback monitoring data"
        echo "  all         - Rollback everything (default)"
        echo ""
        echo "Parameters:"
        echo "  BACKUP_PATH - Path to backup directory or file"
        echo "  FORCE       - Force overwrite existing files (default: false)"
        echo ""
        echo "Examples:"
        echo "  $0 all backups/20240101-120000"
        echo "  $0 data backups/20240101-120000 true"
        echo "  $0 config backups/20240101-120000"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
