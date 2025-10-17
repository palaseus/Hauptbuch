#!/bin/bash

# Hauptbuch Validation Script
# This script validates the Hauptbuch installation and configuration

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
VALIDATION_TYPE=${1:-all}
VERBOSE=${2:-false}

# Validation results
VALIDATION_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Add validation result
add_result() {
    local check_name=$1
    local status=$2
    local message=$3
    
    VALIDATION_RESULTS+=("$check_name:$status:$message")
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$status" = "PASS" ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "$check_name: $message"
    elif [ "$status" = "FAIL" ]; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        print_error "$check_name: $message"
    else
        print_warning "$check_name: $message"
    fi
}

# Validate system requirements
validate_system() {
    print_status "Validating system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        add_result "OS" "PASS" "Supported operating system"
    else
        add_result "OS" "WARN" "Unsupported operating system: $OSTYPE"
    fi
    
    # Check CPU
    local cpu_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "1")
    if [ "$cpu_count" -ge 2 ]; then
        add_result "CPU" "PASS" "Sufficient CPU cores: $cpu_count"
    else
        add_result "CPU" "WARN" "Low CPU core count: $cpu_count"
    fi
    
    # Check memory
    local memory_gb=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo "0")
    if [ "$memory_gb" -ge 4 ]; then
        add_result "Memory" "PASS" "Sufficient memory: ${memory_gb}GB"
    else
        add_result "Memory" "WARN" "Low memory: ${memory_gb}GB (recommended: 4GB+)"
    fi
    
    # Check disk space
    local disk_gb=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$disk_gb" -ge 10 ]; then
        add_result "Disk" "PASS" "Sufficient disk space: ${disk_gb}GB"
    else
        add_result "Disk" "WARN" "Low disk space: ${disk_gb}GB (recommended: 10GB+)"
    fi
}

# Validate Rust installation
validate_rust() {
    print_status "Validating Rust installation..."
    
    # Check Rust version
    if command -v rustc &> /dev/null; then
        local rust_version=$(rustc --version | cut -d' ' -f2)
        add_result "Rust" "PASS" "Rust installed: $rust_version"
        
        # Check Rust components
        if rustup component list --installed | grep -q rustfmt; then
            add_result "Rustfmt" "PASS" "Rustfmt component installed"
        else
            add_result "Rustfmt" "FAIL" "Rustfmt component missing"
        fi
        
        if rustup component list --installed | grep -q clippy; then
            add_result "Clippy" "PASS" "Clippy component installed"
        else
            add_result "Clippy" "FAIL" "Clippy component missing"
        fi
    else
        add_result "Rust" "FAIL" "Rust not installed"
    fi
    
    # Check Cargo
    if command -v cargo &> /dev/null; then
        local cargo_version=$(cargo --version | cut -d' ' -f2)
        add_result "Cargo" "PASS" "Cargo installed: $cargo_version"
    else
        add_result "Cargo" "FAIL" "Cargo not installed"
    fi
}

# Validate dependencies
validate_dependencies() {
    print_status "Validating dependencies..."
    
    # Check Git
    if command -v git &> /dev/null; then
        local git_version=$(git --version | cut -d' ' -f3)
        add_result "Git" "PASS" "Git installed: $git_version"
    else
        add_result "Git" "FAIL" "Git not installed"
    fi
    
    # Check Docker
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        add_result "Docker" "PASS" "Docker installed: $docker_version"
        
        # Check Docker daemon
        if docker ps > /dev/null 2>&1; then
            add_result "Docker Daemon" "PASS" "Docker daemon running"
        else
            add_result "Docker Daemon" "FAIL" "Docker daemon not running"
        fi
    else
        add_result "Docker" "WARN" "Docker not installed (optional)"
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        local compose_version=$(docker-compose --version | cut -d' ' -f3 | cut -d',' -f1)
        add_result "Docker Compose" "PASS" "Docker Compose installed: $compose_version"
    else
        add_result "Docker Compose" "WARN" "Docker Compose not installed (optional)"
    fi
}

# Validate project structure
validate_project() {
    print_status "Validating project structure..."
    
    # Check essential files
    local essential_files=("Cargo.toml" "src/lib.rs" "README.md")
    for file in "${essential_files[@]}"; do
        if [ -f "$file" ]; then
            add_result "File: $file" "PASS" "File exists"
        else
            add_result "File: $file" "FAIL" "File missing"
        fi
    done
    
    # Check directories
    local essential_dirs=("src" "scripts" "monitoring")
    for dir in "${essential_dirs[@]}"; do
        if [ -d "$dir" ]; then
            add_result "Directory: $dir" "PASS" "Directory exists"
        else
            add_result "Directory: $dir" "FAIL" "Directory missing"
        fi
    done
    
    # Check configuration files
    if [ -f "config.toml" ]; then
        add_result "Config" "PASS" "Configuration file exists"
    else
        add_result "Config" "WARN" "Configuration file missing"
    fi
    
    if [ -f ".env" ]; then
        add_result "Environment" "PASS" "Environment file exists"
    else
        add_result "Environment" "WARN" "Environment file missing"
    fi
}

# Validate build
validate_build() {
    print_status "Validating build..."
    
    # Check if project builds
    if cargo check --quiet 2>/dev/null; then
        add_result "Build" "PASS" "Project builds successfully"
    else
        add_result "Build" "FAIL" "Project build failed"
    fi
    
    # Check if tests pass
    if cargo test --quiet 2>/dev/null; then
        add_result "Tests" "PASS" "All tests pass"
    else
        add_result "Tests" "FAIL" "Some tests fail"
    fi
    
    # Check if clippy passes
    if cargo clippy --quiet 2>/dev/null; then
        add_result "Clippy" "PASS" "Clippy checks pass"
    else
        add_result "Clippy" "WARN" "Clippy warnings found"
    fi
}

# Validate services
validate_services() {
    print_status "Validating services..."
    
    # Check if Docker containers are running
    if command -v docker &> /dev/null; then
        local containers=$(docker ps --filter "name=hauptbuch" --format "{{.Names}}" | wc -l)
        if [ "$containers" -gt 0 ]; then
            add_result "Docker Containers" "PASS" "$containers Hauptbuch containers running"
        else
            add_result "Docker Containers" "WARN" "No Hauptbuch containers running"
        fi
    fi
    
    # Check network ports
    local ports=("30303" "8545" "8546" "6379" "9090" "3000")
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            add_result "Port: $port" "PASS" "Port $port is listening"
        else
            add_result "Port: $port" "WARN" "Port $port is not listening"
        fi
    done
}

# Validate security
validate_security() {
    print_status "Validating security..."
    
    # Check file permissions
    if [ -f "config.toml" ] && [ -r "config.toml" ]; then
        add_result "Config Permissions" "PASS" "Configuration file readable"
    else
        add_result "Config Permissions" "WARN" "Configuration file not readable"
    fi
    
    # Check for sensitive files
    if [ -f ".env" ]; then
        local env_perms=$(stat -c %a .env 2>/dev/null || stat -f %A .env 2>/dev/null || echo "000")
        if [ "$env_perms" -le 644 ]; then
            add_result "Environment Permissions" "PASS" "Environment file permissions secure"
        else
            add_result "Environment Permissions" "WARN" "Environment file permissions too open"
        fi
    fi
    
    # Check for backup files
    local backup_files=$(find . -name "*.backup" -o -name "*.bak" | wc -l)
    if [ "$backup_files" -eq 0 ]; then
        add_result "Backup Files" "PASS" "No backup files found"
    else
        add_result "Backup Files" "WARN" "$backup_files backup files found"
    fi
}

# Validate all
validate_all() {
    print_status "🔍 Starting comprehensive validation..."
    
    validate_system
    validate_rust
    validate_dependencies
    validate_project
    validate_build
    validate_services
    validate_security
    
    print_status "Validation completed!"
}

# Generate validation report
generate_report() {
    local report_file="validation-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating validation report: $report_file"
    
    {
        echo "Hauptbuch Validation Report"
        echo "=========================="
        echo "Generated: $(date)"
        echo "Total Checks: $TOTAL_CHECKS"
        echo "Passed: $PASSED_CHECKS"
        echo "Failed: $FAILED_CHECKS"
        echo "Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
        echo ""
        
        echo "Detailed Results:"
        echo "================"
        for result in "${VALIDATION_RESULTS[@]}"; do
            IFS=':' read -r name status message <<< "$result"
            echo "$name: $status - $message"
        done
        
    } > "$report_file"
    
    print_success "Validation report generated: $report_file"
}

# Show validation summary
show_summary() {
    print_status "Validation Summary:"
    echo "  Total Checks: $TOTAL_CHECKS"
    echo "  Passed: $PASSED_CHECKS"
    echo "  Failed: $FAILED_CHECKS"
    echo "  Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
    echo ""
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        print_success "✅ All validations passed!"
    else
        print_error "❌ $FAILED_CHECKS validations failed!"
    fi
}

# Main validation function
main() {
    print_status "🔍 Starting Hauptbuch validation..."
    
    case $VALIDATION_TYPE in
        "system")
            validate_system
            ;;
        "rust")
            validate_rust
            ;;
        "dependencies")
            validate_dependencies
            ;;
        "project")
            validate_project
            ;;
        "build")
            validate_build
            ;;
        "services")
            validate_services
            ;;
        "security")
            validate_security
            ;;
        "all")
            validate_all
            ;;
        *)
            print_error "Unknown validation type: $VALIDATION_TYPE"
            echo "Valid types: system, rust, dependencies, project, build, services, security, all"
            exit 1
            ;;
    esac
    
    show_summary
    
    if [ "$VERBOSE" = "true" ]; then
        generate_report
    fi
    
    # Exit with error code if any validations failed
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        exit 1
    fi
}

# Handle command line arguments
case "${1:-help}" in
    "system"|"rust"|"dependencies"|"project"|"build"|"services"|"security"|"all")
        main "$@"
        ;;
    "report")
        generate_report
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Validation Script"
        echo ""
        echo "Usage: $0 [VALIDATION_TYPE] [VERBOSE]"
        echo ""
        echo "Validation Types:"
        echo "  system       - System requirements"
        echo "  rust         - Rust installation"
        echo "  dependencies - System dependencies"
        echo "  project      - Project structure"
        echo "  build        - Build validation"
        echo "  services     - Service validation"
        echo "  security     - Security validation"
        echo "  all          - All validations (default)"
        echo ""
        echo "Commands:"
        echo "  report       - Generate validation report"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  VERBOSE      - Generate detailed report (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
