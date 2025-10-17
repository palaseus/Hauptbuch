#!/bin/bash

# Hauptbuch Backup Script
# This script creates backups of Hauptbuch data and configuration

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
BACKUP_TYPE=${1:-all}
BACKUP_DIR="backups/$(date +%Y%m%d-%H%M%S)"
COMPRESS=${2:-true}
ENCRYPT=${3:-false}

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup data directory
backup_data() {
    print_status "Backing up data directory..."
    
    if [ -d "data" ]; then
        if [ "$COMPRESS" = "true" ]; then
            tar -czf "$BACKUP_DIR/data.tar.gz" data/
            print_success "Data directory backed up: $BACKUP_DIR/data.tar.gz"
        else
            cp -r data "$BACKUP_DIR/"
            print_success "Data directory backed up: $BACKUP_DIR/data/"
        fi
    else
        print_warning "No data directory found"
    fi
}

# Backup configuration
backup_config() {
    print_status "Backing up configuration..."
    
    # Backup config files
    cp config.toml "$BACKUP_DIR/" 2>/dev/null || print_warning "config.toml not found"
    cp .env "$BACKUP_DIR/" 2>/dev/null || print_warning ".env not found"
    cp env.example "$BACKUP_DIR/" 2>/dev/null || print_warning "env.example not found"
    
    # Backup Docker configuration
    cp docker-compose.yml "$BACKUP_DIR/" 2>/dev/null || print_warning "docker-compose.yml not found"
    cp Dockerfile "$BACKUP_DIR/" 2>/dev/null || print_warning "Dockerfile not found"
    
    # Backup monitoring configuration
    if [ -d "monitoring" ]; then
        cp -r monitoring "$BACKUP_DIR/"
        print_success "Monitoring configuration backed up"
    fi
    
    print_success "Configuration backed up"
}

# Backup logs
backup_logs() {
    print_status "Backing up logs..."
    
    if [ -d "logs" ]; then
        if [ "$COMPRESS" = "true" ]; then
            tar -czf "$BACKUP_DIR/logs.tar.gz" logs/
            print_success "Logs backed up: $BACKUP_DIR/logs.tar.gz"
        else
            cp -r logs "$BACKUP_DIR/"
            print_success "Logs backed up: $BACKUP_DIR/logs/"
        fi
    else
        print_warning "No logs directory found"
    fi
}

# Backup Docker volumes
backup_docker_volumes() {
    print_status "Backing up Docker volumes..."
    
    if ! command -v docker &> /dev/null; then
        print_warning "Docker is not installed"
        return 0
    fi
    
    # List Hauptbuch volumes
    local volumes=$(docker volume ls --filter "name=hauptbuch" --format "{{.Name}}")
    
    if [ -n "$volumes" ]; then
        for volume in $volumes; do
            print_status "Backing up volume: $volume"
            docker run --rm -v "$volume":/data -v "$(pwd)/$BACKUP_DIR":/backup alpine tar czf "/backup/$volume.tar.gz" -C /data .
        done
        print_success "Docker volumes backed up"
    else
        print_warning "No Hauptbuch volumes found"
    fi
}

# Backup source code
backup_source() {
    print_status "Backing up source code..."
    
    # Create source backup
    if [ "$COMPRESS" = "true" ]; then
        tar -czf "$BACKUP_DIR/source.tar.gz" \
            --exclude=target \
            --exclude=node_modules \
            --exclude=.git \
            --exclude=data \
            --exclude=logs \
            --exclude=backups \
            .
        print_success "Source code backed up: $BACKUP_DIR/source.tar.gz"
    else
        cp -r . "$BACKUP_DIR/source/"
        print_success "Source code backed up: $BACKUP_DIR/source/"
    fi
}

# Backup database
backup_database() {
    print_status "Backing up database..."
    
    # Backup RocksDB
    if [ -d "data" ]; then
        if [ "$COMPRESS" = "true" ]; then
            tar -czf "$BACKUP_DIR/database.tar.gz" data/
            print_success "Database backed up: $BACKUP_DIR/database.tar.gz"
        else
            cp -r data "$BACKUP_DIR/database/"
            print_success "Database backed up: $BACKUP_DIR/database/"
        fi
    else
        print_warning "No database found"
    fi
}

# Backup monitoring data
backup_monitoring() {
    print_status "Backing up monitoring data..."
    
    # Backup Prometheus data
    if [ -d "monitoring/data/prometheus" ]; then
        if [ "$COMPRESS" = "true" ]; then
            tar -czf "$BACKUP_DIR/prometheus-data.tar.gz" monitoring/data/prometheus/
            print_success "Prometheus data backed up: $BACKUP_DIR/prometheus-data.tar.gz"
        else
            cp -r monitoring/data/prometheus "$BACKUP_DIR/"
            print_success "Prometheus data backed up"
        fi
    fi
    
    # Backup Grafana data
    if [ -d "monitoring/data/grafana" ]; then
        if [ "$COMPRESS" = "true" ]; then
            tar -czf "$BACKUP_DIR/grafana-data.tar.gz" monitoring/data/grafana/
            print_success "Grafana data backed up: $BACKUP_DIR/grafana-data.tar.gz"
        else
            cp -r monitoring/data/grafana "$BACKUP_DIR/"
            print_success "Grafana data backed up"
        fi
    fi
}

# Encrypt backup
encrypt_backup() {
    if [ "$ENCRYPT" = "true" ]; then
        print_status "Encrypting backup..."
        
        # Create encryption key if it doesn't exist
        if [ ! -f "backup.key" ]; then
            openssl rand -base64 32 > backup.key
            print_warning "Backup encryption key created: backup.key"
            print_warning "Keep this key safe! You'll need it to restore backups."
        fi
        
        # Encrypt backup directory
        tar -czf - "$BACKUP_DIR" | openssl enc -aes-256-cbc -salt -in - -out "$BACKUP_DIR.enc" -pass file:backup.key
        
        # Remove unencrypted backup
        rm -rf "$BACKUP_DIR"
        
        print_success "Backup encrypted: $BACKUP_DIR.enc"
    fi
}

# Create backup manifest
create_manifest() {
    print_status "Creating backup manifest..."
    
    local manifest_file="$BACKUP_DIR/manifest.txt"
    
    {
        echo "Hauptbuch Backup Manifest"
        echo "========================="
        echo "Created: $(date)"
        echo "Type: $BACKUP_TYPE"
        echo "Compressed: $COMPRESS"
        echo "Encrypted: $ENCRYPT"
        echo ""
        echo "Contents:"
        ls -la "$BACKUP_DIR"
        echo ""
        echo "System Information:"
        echo "OS: $(uname -a)"
        echo "Disk Usage: $(df -h . | tail -1)"
        echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
        
    } > "$manifest_file"
    
    print_success "Backup manifest created: $manifest_file"
}

# Backup all
backup_all() {
    print_status "Creating complete backup..."
    
    backup_data
    backup_config
    backup_logs
    backup_docker_volumes
    backup_source
    backup_database
    backup_monitoring
    
    create_manifest
    encrypt_backup
    
    print_success "Complete backup created: $BACKUP_DIR"
}

# List existing backups
list_backups() {
    print_status "Existing backups:"
    
    if [ -d "backups" ]; then
        ls -la backups/
    else
        print_warning "No backups directory found"
    fi
}

# Restore backup
restore_backup() {
    local backup_path=$1
    
    if [ -z "$backup_path" ]; then
        print_error "Backup path not specified"
        exit 1
    fi
    
    print_status "Restoring backup: $backup_path"
    
    # Check if backup exists
    if [ ! -e "$backup_path" ]; then
        print_error "Backup not found: $backup_path"
        exit 1
    fi
    
    # Extract backup
    if [ -f "$backup_path" ]; then
        tar -xzf "$backup_path" -C .
        print_success "Backup restored from: $backup_path"
    else
        print_error "Invalid backup format: $backup_path"
        exit 1
    fi
}

# Show backup information
show_backup_info() {
    print_status "Backup information:"
    echo "  Directory: $BACKUP_DIR"
    echo "  Type: $BACKUP_TYPE"
    echo "  Compressed: $COMPRESS"
    echo "  Encrypted: $ENCRYPT"
    echo ""
    
    if [ -d "$BACKUP_DIR" ]; then
        echo "Contents:"
        ls -la "$BACKUP_DIR"
        echo ""
        echo "Size: $(du -sh "$BACKUP_DIR" | cut -f1)"
    fi
}

# Main backup function
main() {
    print_status "💾 Starting Hauptbuch backup..."
    
    case $BACKUP_TYPE in
        "data")
            backup_data
            ;;
        "config")
            backup_config
            ;;
        "logs")
            backup_logs
            ;;
        "docker")
            backup_docker_volumes
            ;;
        "source")
            backup_source
            ;;
        "database")
            backup_database
            ;;
        "monitoring")
            backup_monitoring
            ;;
        "all")
            backup_all
            ;;
        *)
            print_error "Unknown backup type: $BACKUP_TYPE"
            echo "Valid types: data, config, logs, docker, source, database, monitoring, all"
            exit 1
            ;;
    esac
    
    show_backup_info
    print_success "✅ Backup completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "data"|"config"|"logs"|"docker"|"source"|"database"|"monitoring"|"all")
        main "$@"
        ;;
    "list")
        list_backups
        ;;
    "restore")
        restore_backup "$2"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Backup Script"
        echo ""
        echo "Usage: $0 [BACKUP_TYPE] [COMPRESS] [ENCRYPT]"
        echo ""
        echo "Backup Types:"
        echo "  data        - Backup data directory"
        echo "  config      - Backup configuration files"
        echo "  logs        - Backup log files"
        echo "  docker      - Backup Docker volumes"
        echo "  source      - Backup source code"
        echo "  database    - Backup database"
        echo "  monitoring  - Backup monitoring data"
        echo "  all         - Backup everything (default)"
        echo ""
        echo "Commands:"
        echo "  list        - List existing backups"
        echo "  restore     - Restore from backup"
        echo "  help        - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  COMPRESS    - Compress backup (default: true)"
        echo "  ENCRYPT     - Encrypt backup (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac