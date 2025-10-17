# Hauptbuch Scripts

This directory contains various scripts to help with Hauptbuch development, deployment, and maintenance.

## Scripts Overview

### Setup and Installation
- **`setup.sh`** - Initial project setup and dependency installation
- **`install-deps.sh`** - Install system dependencies and tools
- **`quick-start.sh`** - Quick start guide for new users
- **`install.sh`** - Install Hauptbuch on the system
- **`uninstall.sh`** - Uninstall Hauptbuch from the system

### Development
- **`dev.sh`** - Development workflow commands
- **`test.sh`** - Comprehensive testing suite
- **`benchmark.sh`** - Performance benchmarking
- **`clean.sh`** - Cleanup build artifacts and temporary files

### Deployment and Operations
- **`deploy.sh`** - Deploy to various environments
- **`update.sh`** - Update Hauptbuch to latest version
- **`upgrade.sh`** - Upgrade Hauptbuch to latest version
- **`backup.sh`** - Backup data and configuration
- **`restore.sh`** - Restore from backup
- **`rollback.sh`** - Rollback to previous version
- **`health-check.sh`** - Check service health and status
- **`status.sh`** - Show system and service status

### Monitoring and Logging
- **`monitor.sh`** - Monitor system and service metrics
- **`logs.sh`** - Manage and analyze logs
- **`network.sh`** - Network configuration and monitoring
- **`performance.sh`** - Performance analysis and optimization
- **`security.sh`** - Security checks and hardening
- **`validate.sh`** - Validate installation and configuration

## Usage

### Quick Start
```bash
# Get started quickly
./scripts/quick-start.sh

# Development mode
./scripts/quick-start.sh development

# Production mode
./scripts/quick-start.sh production

# Testing mode
./scripts/quick-start.sh testing
```

### Development Workflow
```bash
# Start development server
./scripts/dev.sh dev

# Run tests
./scripts/dev.sh test

# Run benchmarks
./scripts/dev.sh bench

# Format code
./scripts/dev.sh fmt

# Run clippy
./scripts/dev.sh clippy

# Start Docker services
./scripts/dev.sh docker

# Stop services
./scripts/dev.sh stop

# View logs
./scripts/dev.sh logs

# Open monitoring
./scripts/dev.sh monitor
```

### Testing
```bash
# Run all tests
./scripts/test.sh all

# Run specific test types
./scripts/test.sh unit
./scripts/test.sh integration
./scripts/test.sh performance
./scripts/test.sh security
./scripts/test.sh chaos
./scripts/test.sh fuzzing
./scripts.test.sh property

# Run with coverage
./scripts/test.sh all true

# Run in parallel
./scripts/test.sh all false true
```

### Benchmarking
```bash
# Run all benchmarks
./scripts/benchmark.sh all

# Run specific benchmarks
./scripts/benchmark.sh consensus
./scripts/benchmark.sh crypto
./scripts/benchmark.sh performance
./scripts/benchmark.sh network
./scripts/benchmark.sh l2
./scripts/benchmark.sh security
./scripts/benchmark.sh stress
```

### Deployment
```bash
# Deploy to development
./scripts/deploy.sh development

# Deploy to staging
./scripts/deploy.sh staging

# Deploy to production
./scripts/deploy.sh production

# Show deployment status
./scripts/deploy.sh status

# Rollback deployment
./scripts/deploy.sh rollback
```

### Maintenance
```bash
# Update Hauptbuch
./scripts/update.sh all

# Upgrade Hauptbuch
./scripts/upgrade.sh all

# Update specific components
./scripts/update.sh rust
./scripts/update.sh dependencies
./scripts/update.sh source
./scripts/update.sh docker
./scripts/update.sh config
./scripts/update.sh monitoring

# Show update status
./scripts/update.sh status

# Rollback update
./scripts/update.sh rollback
```

### Backup and Restore
```bash
# Create backup
./scripts/backup.sh all

# Backup specific components
./scripts/backup.sh data
./scripts/backup.sh config
./scripts/backup.sh logs
./scripts/backup.sh docker
./scripts/backup.sh source
./scripts/backup.sh database
./scripts/backup.sh monitoring

# List backups
./scripts/backup.sh list

# Restore from backup
./scripts/backup.sh restore backups/20240101-120000

# Rollback to previous version
./scripts/rollback.sh all backups/20240101-120000
```

### Health and Status
```bash
# Check health
./scripts/health-check.sh

# Show status
./scripts/status.sh all

# Show specific status
./scripts/status.sh system
./scripts/status.sh docker
./scripts/status.sh network
./scripts/status.sh services
./scripts/status.sh project
./scripts/status.sh config
./scripts/status.sh monitoring
./scripts/status.sh performance
./scripts/status.sh security

# Generate status report
./scripts/status.sh report
```

### Monitoring
```bash
# Monitor all metrics
./scripts/monitor.sh all

# Monitor specific metrics
./scripts/monitor.sh system
./scripts/monitor.sh docker
./scripts/monitor.sh network
./scripts/monitor.sh services
./scripts/monitor.sh application

# Continuous monitoring
./scripts/monitor.sh continuous

# Generate monitoring report
./scripts/monitor.sh report
```

### Logs
```bash
# Show all logs
./scripts/logs.sh all

# Show specific logs
./scripts/logs.sh node
./scripts/logs.sh redis
./scripts/logs.sh prometheus
./scripts/logs.sh grafana
./scripts/logs.sh system
./scripts/logs.sh application

# Follow logs
./scripts/logs.sh follow

# Filter logs by level
./scripts/logs.sh filter error

# Search logs
./scripts/logs.sh search "error"

# Analyze logs
./scripts/logs.sh analyze

# Export logs
./scripts/logs.sh export

# Clean logs
./scripts/logs.sh clean

# Rotate logs
./scripts/logs.sh rotate

# Show log statistics
./scripts/logs.sh stats
```

### Network
```bash
# Check network connectivity
./scripts/network.sh connectivity

# Check network interfaces
./scripts/network.sh interfaces

# Check network ports
./scripts/network.sh ports

# Check network security
./scripts/network.sh security

# Check network performance
./scripts/network.sh performance

# Check all network
./scripts/network.sh all

# Configure network
./scripts/network.sh configure

# Monitor network
./scripts/network.sh monitor

# Generate network report
./scripts/network.sh report
```

### Performance
```bash
# Check system performance
./scripts/performance.sh system

# Check build performance
./scripts/performance.sh build

# Check test performance
./scripts/performance.sh test

# Check runtime performance
./scripts/performance.sh runtime

# Check network performance
./scripts/performance.sh network

# Check database performance
./scripts/performance.sh database

# Check all performance
./scripts/performance.sh all

# Optimize performance
./scripts/performance.sh optimize

# Profile performance
./scripts/performance.sh profile

# Generate performance report
./scripts/performance.sh report
```

### Security
```bash
# Check file security
./scripts/security.sh files

# Check data security
./scripts/security.sh data

# Check network security
./scripts/security.sh network

# Check Docker security
./scripts/security.sh docker

# Check Rust security
./scripts/security.sh rust

# Check dependencies security
./scripts/security.sh dependencies

# Check configuration security
./scripts/security.sh config

# Check all security
./scripts/security.sh all

# Fix security issues
./scripts/security.sh fix

# Generate security report
./scripts/security.sh report
```

### Validation
```bash
# Validate system requirements
./scripts/validate.sh system

# Validate Rust installation
./scripts/validate.sh rust

# Validate dependencies
./scripts/validate.sh dependencies

# Validate project structure
./scripts/validate.sh project

# Validate build
./scripts/validate.sh build

# Validate services
./scripts/validate.sh services

# Validate security
./scripts/validate.sh security

# Validate all
./scripts/validate.sh all

# Generate validation report
./scripts/validate.sh report
```

### Cleanup
```bash
# Clean everything
./scripts/clean.sh all

# Clean specific components
./scripts/clean.sh build
./scripts/clean.sh temp
./scripts/clean.sh data
./scripts/clean.sh docker
./scripts/clean.sh monitoring
./scripts/clean.sh cache

# Force cleanup
./scripts/clean.sh all true
```

### Installation
```bash
# Install everything
./scripts/install.sh all

# Install specific components
./scripts/install.sh dependencies
./scripts/install.sh rust
./scripts/install.sh user
./scripts/install.sh directories
./scripts/install.sh hauptbuch
./scripts/install.sh service
./scripts/install.sh firewall
./scripts/install.sh monitoring

# Show installation status
./scripts/install.sh status
```

### Uninstallation
```bash
# Uninstall everything
./scripts/uninstall.sh all

# Uninstall specific components
./scripts/uninstall.sh services
./scripts/uninstall.sh hauptbuch
./scripts/uninstall.sh monitoring
./scripts/uninstall.sh data
./scripts/uninstall.sh directories
./scripts/uninstall.sh user
./scripts/uninstall.sh docker
./scripts/uninstall.sh dependencies
./scripts/uninstall.sh rust

# Show uninstallation status
./scripts/uninstall.sh status
```

## Configuration

### Environment Variables
Set these environment variables to customize script behavior:

```bash
# Development
export HAUPTBUCH_ENVIRONMENT=development
export HAUPTBUCH_LOG_LEVEL=info

# Production
export HAUPTBUCH_ENVIRONMENT=production
export HAUPTBUCH_LOG_LEVEL=warn

# Testing
export HAUPTBUCH_ENVIRONMENT=testing
export HAUPTBUCH_LOG_LEVEL=debug
```

### Script Configuration
Each script can be configured using command line arguments:

```bash
# Show help for any script
./scripts/script-name.sh help

# Example: Show dev.sh help
./scripts/dev.sh help
```

## Best Practices

### Development
1. Always run tests before committing
2. Use `./scripts/dev.sh fmt` to format code
3. Use `./scripts/dev.sh clippy` to check for issues
4. Use `./scripts/test.sh all` to run comprehensive tests

### Deployment
1. Always backup before deployment
2. Test in staging environment first
3. Use `./scripts/health-check.sh` to verify deployment
4. Monitor services after deployment

### Maintenance
1. Regular backups using `./scripts/backup.sh`
2. Regular updates using `./scripts/update.sh`
3. Regular health checks using `./scripts/health-check.sh`
4. Regular cleanup using `./scripts/clean.sh`

## Troubleshooting

### Common Issues

#### Script Permission Denied
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

#### Docker Not Running
```bash
# Start Docker daemon
sudo systemctl start docker

# Check Docker status
docker ps
```

#### Port Already in Use
```bash
# Check what's using the port
lsof -i :8545

# Kill the process
sudo kill -9 <PID>
```

#### Build Failures
```bash
# Clean and rebuild
./scripts/clean.sh build
cargo build
```

#### Test Failures
```bash
# Run tests with verbose output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

### Getting Help
1. Check script help: `./scripts/script-name.sh help`
2. Check logs: `./scripts/dev.sh logs`
3. Check status: `./scripts/status.sh all`
4. Check health: `./scripts/health-check.sh`

## Contributing

When adding new scripts:

1. Follow the existing naming convention
2. Include comprehensive help text
3. Add error handling and validation
4. Use consistent output formatting
5. Add to this README

## License

These scripts are part of the Hauptbuch project and follow the same license terms.