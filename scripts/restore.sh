#!/bin/bash

# Hauptbuch Restore Script
# This script restores Hauptbuch from backups

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
RESTORE_TYPE=${1:-all}
BACKUP_PATH=${2:-""}
FORCE=${3:-false}

# Validate backup path
validate_backup_path() {
    if [ -z "$BACKUP_PATH" ]; then
        print_error "Backup path not specified"
        exit 1
    fi
    
    if [ ! -e "$BACKUP_PATH" ]; then
        print_error "Backup not found: $BACKUP_PATH"
        exit 1
    fi
    
    print_success "Backup path validated: $BACKUP_PATH"
}

# Restore data directory
restore_data() {
    print_status "Restoring data directory..."
    
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
        print_success "Data directory restored from: $BACKUP_PATH/data.tar.gz"
    elif [ -d "$BACKUP_PATH/data" ]; then
        cp -r "$BACKUP_PATH/data" .
        print_success "Data directory restored from: $BACKUP_PATH/data/"
    else
        print_warning "No data backup found in: $BACKUP_PATH"
    fi
}

# Restore configuration
restore_config() {
    print_status "Restoring configuration..."
    
    # Restore config files
    if [ -f "$BACKUP_PATH/config.toml" ]; then
        cp "$BACKUP_PATH/config.toml" .
        print_success "config.toml restored"
    fi
    
    if [ -f "$BACKUP_PATH/.env" ]; then
        cp "$BACKUP_PATH/.env" .
        print_success ".env restored"
    fi
    
    if [ -f "$BACKUP_PATH/env.example" ]; then
        cp "$BACKUP_PATH/env.example" .
        print_success "env.example restored"
    fi
    
    # Restore Docker configuration
    if [ -f "$BACKUP_PATH/docker-compose.yml" ]; then
        cp "$BACKUP_PATH/docker-compose.yml" .
        print_success "docker-compose.yml restored"
    fi
    
    if [ -f "$BACKUP_PATH/Dockerfile" ]; then
        cp "$BACKUP_PATH/Dockerfile" .
        print_success "Dockerfile restored"
    fi
    
    # Restore monitoring configuration
    if [ -d "$BACKUP_PATH/monitoring" ]; then
        cp -r "$BACKUP_PATH/monitoring" .
        print_success "Monitoring configuration restored"
    fi
}

# Restore logs
restore_logs() {
    print_status "Restoring logs..."
    
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
        print_success "Logs restored from: $BACKUP_PATH/logs.tar.gz"
    elif [ -d "$BACKUP_PATH/logs" ]; then
        cp -r "$BACKUP_PATH/logs" .
        print_success "Logs restored from: $BACKUP_PATH/logs/"
    else
        print_warning "No logs backup found in: $BACKUP_PATH"
    fi
}

# Restore Docker volumes
restore_docker_volumes() {
    print_status "Restoring Docker volumes..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    # List Hauptbuch volumes
    local volumes=$(docker volume ls --filter "name=hauptbuch" --format "{{.Name}}")
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            if [ -f "$BACKUP_PATH/$volume.tar.gz" ]; then
                print_status "Restoring volume: $volume"
                docker run --rm -v "$volume":/data -v "$(pwd)/$BACKUP_PATH":/backup alpine tar xzf "/backup/$volume.tar.gz" -C /data
                print_success "Volume $volume restored"
            else
                print_warning "No backup found for volume: $volume"
            fi
        done
    else
        print_warning "No Hauptbuch volumes found"
    fi
}

# Restore source code
restore_source() {
    print_status "Restoring source code..."
    
    # Check if source directory exists
    if [ -d "src" ] && [ "$FORCE" != "true" ]; then
        print_warning "Source directory already exists. Use --force to overwrite."
        return 0
    fi
    
    # Restore from backup
    if [ -f "$BACKUP_PATH/source.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/source.tar.gz"
        print_success "Source code restored from: $BACKUP_PATH/source.tar.gz"
    elif [ -d "$BACKUP_PATH/source" ]; then
        cp -r "$BACKUP_PATH/source"/* .
        print_success "Source code restored from: $BACKUP_PATH/source/"
    else
        print_warning "No source backup found in: $BACKUP_PATH"
    fi
}

# Restore database
restore_database() {
    print_status "Restoring database..."
    
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
        print_success "Database restored from: $BACKUP_PATH/database.tar.gz"
    elif [ -d "$BACKUP_PATH/database" ]; then
        cp -r "$BACKUP_PATH/database" data
        print_success "Database restored from: $BACKUP_PATH/database/"
    else
        print_warning "No database backup found in: $BACKUP_PATH"
    fi
}

# Restore monitoring data
restore_monitoring() {
    print_status "Restoring monitoring data..."
    
    # Create monitoring directory
    mkdir -p monitoring/data
    
    # Restore Prometheus data
    if [ -f "$BACKUP_PATH/prometheus-data.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/prometheus-data.tar.gz" -C monitoring/data/
        print_success "Prometheus data restored from: $BACKUP_PATH/prometheus-data.tar.gz"
    elif [ -d "$BACKUP_PATH/prometheus-data" ]; then
        cp -r "$BACKUP_PATH/prometheus-data" monitoring/data/
        print_success "Prometheus data restored from: $BACKUP_PATH/prometheus-data/"
    else
        print_warning "No Prometheus data backup found in: $BACKUP_PATH"
    fi
    
    # Restore Grafana data
    if [ -f "$BACKUP_PATH/grafana-data.tar.gz" ]; then
        tar -xzf "$BACKUP_PATH/grafana-data.tar.gz" -C monitoring/data/
        print_success "Grafana data restored from: $BACKUP_PATH/grafana-data.tar.gz"
    elif [ -d "$BACKUP_PATH/grafana-data" ]; then
        cp -r "$BACKUP_PATH/grafana-data" monitoring/data/
        print_success "Grafana data restored from: $BACKUP_PATH/grafana-data/"
    else
        print_warning "No Grafana data backup found in: $BACKUP_PATH"
    fi
}

# Decrypt backup
decrypt_backup() {
    if [ -f "$BACKUP_PATH.enc" ]; then
        print_status "Decrypting backup..."
        
        # Check for encryption key
        if [ ! -f "backup.key" ]; then
            print_error "Backup encryption key not found: backup.key"
            exit 1
        fi
        
        # Decrypt backup
        openssl enc -aes-256-cbc -d -in "$BACKUP_PATH.enc" -out "$BACKUP_PATH.tar.gz" -pass file:backup.key
        
        # Extract decrypted backup
        tar -xzf "$BACKUP_PATH.tar.gz"
        
        # Remove temporary files
        rm -f "$BACKUP_PATH.tar.gz"
        
        print_success "Backup decrypted and extracted"
    fi
}

# Restore all
restore_all() {
    print_status "Restoring complete backup..."
    
    # Decrypt backup if needed
    decrypt_backup
    
    # Restore all components
    restore_data
    restore_config
    restore_logs
    restore_docker_volumes
    restore_source
    restore_database
    restore_monitoring
    
    print_success "Complete backup restored from: $BACKUP_PATH"
}

# Verify restore
verify_restore() {
    print_status "Verifying restore..."
    
    # Check restored components
    local components=("data" "config.toml" "docker-compose.yml" "src")
    local restored_count=0
    
    for component in "${components[@]}"; do
        if [ -e "$component" ]; then
            print_success "Component restored: $component"
            restored_count=$((restored_count + 1))
        else
            print_warning "Component not found: $component"
        fi
    done
    
    print_status "Restored components: $restored_count/${#components[@]}"
    
    # Check file permissions
    if [ -f "config.toml" ]; then
        local perms=$(stat -c %a config.toml 2>/dev/null || stat -f %A config.toml 2>/dev/null || echo "000")
        if [ "$perms" -le 644 ]; then
            print_success "Configuration file permissions are secure"
        else
            print_warning "Configuration file permissions are too open: $perms"
        fi
    fi
    
    print_success "Restore verification completed"
}

# Show restore information
show_restore_info() {
    print_status "Restore information:"
    echo "  Backup path: $BACKUP_PATH"
    echo "  Type: $RESTORE_TYPE"
    echo "  Force: $FORCE"
    echo ""
    
    if [ -d "$BACKUP_PATH" ]; then
        echo "Backup contents:"
        ls -la "$BACKUP_PATH"
        echo ""
        echo "Backup size: $(du -sh "$BACKUP_PATH" | cut -f1)"
    fi
}

# Main restore function
main() {
    print_status "🔄 Starting Hauptbuch restore..."
    
    validate_backup_path
    show_restore_info
    
    case $RESTORE_TYPE in
        "data")
            restore_data
            ;;
        "config")
            restore_config
            ;;
        "logs")
            restore_logs
            ;;
        "docker")
            restore_docker_volumes
            ;;
        "source")
            restore_source
            ;;
        "database")
            restore_database
            ;;
        "monitoring")
            restore_monitoring
            ;;
        "all")
            restore_all
            ;;
        *)
            print_error "Unknown restore type: $RESTORE_TYPE"
            echo "Valid types: data, config, logs, docker, source, database, monitoring, all"
            exit 1
            ;;
    esac
    
    verify_restore
    print_success "✅ Restore completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "data"|"config"|"logs"|"docker"|"source"|"database"|"monitoring"|"all")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Restore Script"
        echo ""
        echo "Usage: $0 [RESTORE_TYPE] [BACKUP_PATH] [FORCE]"
        echo ""
        echo "Restore Types:"
        echo "  data        - Restore data directory"
        echo "  config      - Restore configuration files"
        echo "  logs        - Restore log files"
        echo "  docker      - Restore Docker volumes"
        echo "  source      - Restore source code"
        echo "  database    - Restore database"
        echo "  monitoring  - Restore monitoring data"
        echo "  all         - Restore everything (default)"
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
