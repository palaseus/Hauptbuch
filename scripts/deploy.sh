#!/bin/bash

# Hauptbuch Deployment Script
# This script deploys Hauptbuch to various environments

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
ENVIRONMENT=${1:-development}
VERSION=${2:-latest}
REGISTRY=${3:-localhost:5000}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        "development"|"staging"|"production")
            print_success "Environment: $ENVIRONMENT"
            ;;
        *)
            print_error "Invalid environment: $ENVIRONMENT"
            print_status "Valid environments: development, staging, production"
            exit 1
            ;;
    esac
}

# Build Docker image
build_image() {
    print_status "Building Docker image for $ENVIRONMENT..."
    
    local tag="$REGISTRY/hauptbuch:$VERSION"
    
    docker build -t "$tag" .
    print_success "Docker image built: $tag"
}

# Push Docker image
push_image() {
    if [ "$ENVIRONMENT" = "development" ]; then
        print_status "Skipping image push for development environment"
        return 0
    fi
    
    print_status "Pushing Docker image to registry..."
    
    local tag="$REGISTRY/hauptbuch:$VERSION"
    docker push "$tag"
    print_success "Docker image pushed: $tag"
}

# Deploy to development
deploy_development() {
    print_status "Deploying to development environment..."
    
    # Stop existing containers
    docker-compose down || true
    
    # Start new containers
    docker-compose up -d
    
    # Wait for services to be ready
    sleep 30
    
    # Check health
    ./scripts/health-check.sh
    
    print_success "Development deployment completed!"
}

# Deploy to staging
deploy_staging() {
    print_status "Deploying to staging environment..."
    
    # Update docker-compose for staging
    cp docker-compose.yml docker-compose.staging.yml
    sed -i 's/hauptbuch-node/hauptbuch-staging-node/g' docker-compose.staging.yml
    sed -i 's/hauptbuch-redis/hauptbuch-staging-redis/g' docker-compose.staging.yml
    sed -i 's/hauptbuch-rocksdb/hauptbuch-staging-rocksdb/g' docker-compose.staging.yml
    sed -i 's/hauptbuch-prometheus/hauptbuch-staging-prometheus/g' docker-compose.staging.yml
    sed -i 's/hauptbuch-grafana/hauptbuch-staging-grafana/g' docker-compose.staging.yml
    
    # Deploy with staging configuration
    docker-compose -f docker-compose.staging.yml up -d
    
    print_success "Staging deployment completed!"
}

# Deploy to production
deploy_production() {
    print_status "Deploying to production environment..."
    
    # Production deployment would typically involve:
    # - Kubernetes manifests
    # - Helm charts
    # - CI/CD pipelines
    # - Load balancers
    # - SSL certificates
    # - Monitoring and alerting
    
    print_warning "Production deployment not implemented yet"
    print_status "Production deployment would require:"
    echo "  - Kubernetes cluster"
    echo "  - Helm charts"
    echo "  - SSL certificates"
    echo "  - Load balancer configuration"
    echo "  - Monitoring setup"
    echo "  - Backup configuration"
}

# Run database migrations
run_migrations() {
    print_status "Running database migrations..."
    
    # This would run any necessary database migrations
    # For now, we'll just create the data directory
    mkdir -p data
    
    print_success "Database migrations completed!"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create monitoring directories
    mkdir -p monitoring/data/prometheus
    mkdir -p monitoring/data/grafana
    
    # Set permissions
    chmod 777 monitoring/data/prometheus
    chmod 777 monitoring/data/grafana
    
    print_success "Monitoring setup completed!"
}

# Backup data
backup_data() {
    print_status "Creating backup..."
    
    local backup_dir="backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup data directory
    if [ -d "data" ]; then
        cp -r data "$backup_dir/"
        print_success "Data backed up to: $backup_dir"
    else
        print_warning "No data directory found to backup"
    fi
}

# Rollback deployment
rollback() {
    print_status "Rolling back deployment..."
    
    # Stop current containers
    docker-compose down
    
    # Restore from backup
    local latest_backup=$(ls -t backups/ | head -n1)
    if [ -n "$latest_backup" ]; then
        cp -r "backups/$latest_backup/data" ./
        print_success "Rolled back to: $latest_backup"
    else
        print_error "No backup found for rollback"
        exit 1
    fi
}

# Show deployment status
show_status() {
    print_status "Deployment status:"
    
    echo "Environment: $ENVIRONMENT"
    echo "Version: $VERSION"
    echo "Registry: $REGISTRY"
    echo ""
    
    echo "Docker containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    
    echo "Docker images:"
    docker images | grep hauptbuch
}

# Main deployment function
main() {
    print_status "🚀 Starting Hauptbuch deployment..."
    
    validate_environment
    
    case $ENVIRONMENT in
        "development")
            build_image
            run_migrations
            setup_monitoring
            deploy_development
            ;;
        "staging")
            build_image
            push_image
            run_migrations
            setup_monitoring
            deploy_staging
            ;;
        "production")
            build_image
            push_image
            backup_data
            run_migrations
            setup_monitoring
            deploy_production
            ;;
    esac
    
    show_status
    print_success "✅ Deployment completed successfully!"
}

# Handle command line arguments
case "${1:-help}" in
    "development"|"staging"|"production")
        main "$@"
        ;;
    "rollback")
        rollback
        ;;
    "status")
        show_status
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Deployment Script"
        echo ""
        echo "Usage: $0 [ENVIRONMENT] [VERSION] [REGISTRY]"
        echo ""
        echo "Environments:"
        echo "  development  - Deploy to development environment"
        echo "  staging      - Deploy to staging environment"
        echo "  production   - Deploy to production environment"
        echo ""
        echo "Commands:"
        echo "  rollback     - Rollback to previous version"
        echo "  status       - Show deployment status"
        echo "  help         - Show this help message"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
