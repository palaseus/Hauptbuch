#!/bin/bash

# Hauptbuch Security Script
# This script performs security checks and hardening for Hauptbuch

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
SECURITY_TYPE=${1:-all}
AUDIT_LEVEL=${2:-medium}
FIX_ISSUES=${3:-false}

# Security results
SECURITY_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Add security result
add_result() {
    local check_name=$1
    local status=$2
    local message=$3
    local severity=$4
    
    SECURITY_RESULTS+=("$check_name:$status:$message:$severity")
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$status" = "PASS" ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "$check_name: $message"
    elif [ "$status" = "FAIL" ]; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        if [ "$severity" = "HIGH" ]; then
            print_error "$check_name: $message"
        else
            print_warning "$check_name: $message"
        fi
    else
        print_warning "$check_name: $message"
    fi
}

# Check file permissions
check_file_permissions() {
    print_status "Checking file permissions..."
    
    # Check configuration files
    local config_files=("config.toml" ".env" "docker-compose.yml")
    for file in "${config_files[@]}"; do
        if [ -f "$file" ]; then
            local perms=$(stat -c %a "$file" 2>/dev/null || stat -f %A "$file" 2>/dev/null || echo "000")
            if [ "$perms" -le 644 ]; then
                add_result "File Permissions: $file" "PASS" "Secure permissions ($perms)" "MEDIUM"
            else
                add_result "File Permissions: $file" "FAIL" "Insecure permissions ($perms)" "HIGH"
            fi
        fi
    done
    
    # Check script permissions
    local script_files=$(find scripts/ -name "*.sh" 2>/dev/null | wc -l)
    local executable_scripts=$(find scripts/ -name "*.sh" -executable 2>/dev/null | wc -l)
    if [ "$script_files" -eq "$executable_scripts" ]; then
        add_result "Script Permissions" "PASS" "All scripts are executable" "LOW"
    else
        add_result "Script Permissions" "FAIL" "Some scripts are not executable" "MEDIUM"
    fi
}

# Check sensitive data
check_sensitive_data() {
    print_status "Checking for sensitive data..."
    
    # Check for hardcoded secrets
    local secret_patterns=("password" "secret" "key" "token" "api_key")
    for pattern in "${secret_patterns[@]}"; do
        local matches=$(grep -r -i "$pattern" src/ 2>/dev/null | wc -l)
        if [ "$matches" -eq 0 ]; then
            add_result "Hardcoded Secrets: $pattern" "PASS" "No hardcoded secrets found" "HIGH"
        else
            add_result "Hardcoded Secrets: $pattern" "FAIL" "$matches potential secrets found" "HIGH"
        fi
    done
    
    # Check for private keys
    local private_key_files=$(find . -name "*.key" -o -name "*.pem" -o -name "*.p12" 2>/dev/null | wc -l)
    if [ "$private_key_files" -eq 0 ]; then
        add_result "Private Keys" "PASS" "No private key files found" "HIGH"
    else
        add_result "Private Keys" "FAIL" "$private_key_files private key files found" "HIGH"
    fi
    
    # Check for backup files
    local backup_files=$(find . -name "*.backup" -o -name "*.bak" -o -name "*~" 2>/dev/null | wc -l)
    if [ "$backup_files" -eq 0 ]; then
        add_result "Backup Files" "PASS" "No backup files found" "MEDIUM"
    else
        add_result "Backup Files" "FAIL" "$backup_files backup files found" "MEDIUM"
    fi
}

# Check network security
check_network_security() {
    print_status "Checking network security..."
    
    # Check listening ports
    local open_ports=$(netstat -tuln 2>/dev/null | grep LISTEN | wc -l)
    if [ "$open_ports" -le 10 ]; then
        add_result "Open Ports" "PASS" "Reasonable number of open ports ($open_ports)" "MEDIUM"
    else
        add_result "Open Ports" "FAIL" "Too many open ports ($open_ports)" "MEDIUM"
    fi
    
    # Check for exposed services
    local exposed_services=$(netstat -tuln 2>/dev/null | grep -E ":(22|23|21|25|53|80|443|993|995)" | wc -l)
    if [ "$exposed_services" -eq 0 ]; then
        add_result "Exposed Services" "PASS" "No critical services exposed" "HIGH"
    else
        add_result "Exposed Services" "FAIL" "$exposed_services critical services exposed" "HIGH"
    fi
}

# Check Docker security
check_docker_security() {
    print_status "Checking Docker security..."
    
    if ! command -v docker &> /dev/null; then
        add_result "Docker Security" "WARN" "Docker not installed" "LOW"
        return 0
    fi
    
    # Check Docker daemon
    if docker ps > /dev/null 2>&1; then
        add_result "Docker Daemon" "PASS" "Docker daemon accessible" "MEDIUM"
    else
        add_result "Docker Daemon" "FAIL" "Docker daemon not accessible" "MEDIUM"
    fi
    
    # Check for privileged containers
    local privileged_containers=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -i privileged | wc -l)
    if [ "$privileged_containers" -eq 0 ]; then
        add_result "Privileged Containers" "PASS" "No privileged containers running" "HIGH"
    else
        add_result "Privileged Containers" "FAIL" "$privileged_containers privileged containers running" "HIGH"
    fi
    
    # Check for root containers
    local root_containers=$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -i root | wc -l)
    if [ "$root_containers" -eq 0 ]; then
        add_result "Root Containers" "PASS" "No root containers running" "MEDIUM"
    else
        add_result "Root Containers" "FAIL" "$root_containers root containers running" "MEDIUM"
    fi
}

# Check Rust security
check_rust_security() {
    print_status "Checking Rust security..."
    
    # Check for unsafe code
    local unsafe_blocks=$(grep -r "unsafe" src/ 2>/dev/null | wc -l)
    if [ "$unsafe_blocks" -eq 0 ]; then
        add_result "Unsafe Code" "PASS" "No unsafe code found" "HIGH"
    else
        add_result "Unsafe Code" "FAIL" "$unsafe_blocks unsafe blocks found" "HIGH"
    fi
    
    # Check for panic! macros
    local panic_macros=$(grep -r "panic!" src/ 2>/dev/null | wc -l)
    if [ "$panic_macros" -eq 0 ]; then
        add_result "Panic Macros" "PASS" "No panic! macros found" "MEDIUM"
    else
        add_result "Panic Macros" "FAIL" "$panic_macros panic! macros found" "MEDIUM"
    fi
    
    # Check for unwrap() calls
    local unwrap_calls=$(grep -r "unwrap()" src/ 2>/dev/null | wc -l)
    if [ "$unwrap_calls" -le 10 ]; then
        add_result "Unwrap Calls" "PASS" "Reasonable number of unwrap() calls ($unwrap_calls)" "LOW"
    else
        add_result "Unwrap Calls" "FAIL" "Too many unwrap() calls ($unwrap_calls)" "LOW"
    fi
}

# Check dependencies security
check_dependencies_security() {
    print_status "Checking dependencies security..."
    
    # Check for known vulnerabilities
    if command -v cargo &> /dev/null; then
        if cargo audit --quiet 2>/dev/null; then
            add_result "Dependency Vulnerabilities" "PASS" "No known vulnerabilities found" "HIGH"
        else
            add_result "Dependency Vulnerabilities" "FAIL" "Known vulnerabilities found" "HIGH"
        fi
    else
        add_result "Dependency Vulnerabilities" "WARN" "Cargo not available for audit" "MEDIUM"
    fi
    
    # Check for outdated dependencies
    if command -v cargo &> /dev/null; then
        local outdated_deps=$(cargo outdated 2>/dev/null | wc -l)
        if [ "$outdated_deps" -le 5 ]; then
            add_result "Outdated Dependencies" "PASS" "Few outdated dependencies ($outdated_deps)" "MEDIUM"
        else
            add_result "Outdated Dependencies" "FAIL" "Many outdated dependencies ($outdated_deps)" "MEDIUM"
        fi
    fi
}

# Check configuration security
check_configuration_security() {
    print_status "Checking configuration security..."
    
    # Check for default passwords
    if [ -f ".env" ]; then
        local default_passwords=$(grep -i "password.*=" .env | grep -v "PASSWORD=" | wc -l)
        if [ "$default_passwords" -eq 0 ]; then
            add_result "Default Passwords" "PASS" "No default passwords found" "HIGH"
        else
            add_result "Default Passwords" "FAIL" "$default_passwords potential default passwords found" "HIGH"
        fi
    fi
    
    # Check for debug mode in production
    if [ -f "config.toml" ]; then
        if grep -q "debug.*=.*true" config.toml; then
            add_result "Debug Mode" "FAIL" "Debug mode enabled in configuration" "MEDIUM"
        else
            add_result "Debug Mode" "PASS" "Debug mode not enabled" "MEDIUM"
        fi
    fi
    
    # Check for insecure protocols
    if [ -f "config.toml" ]; then
        if grep -q "http://" config.toml; then
            add_result "Insecure Protocols" "FAIL" "HTTP protocol used (should use HTTPS)" "HIGH"
        else
            add_result "Insecure Protocols" "PASS" "No HTTP protocol found" "HIGH"
        fi
    fi
}

# Check all security
check_all_security() {
    print_status "🔒 Starting comprehensive security check..."
    
    check_file_permissions
    check_sensitive_data
    check_network_security
    check_docker_security
    check_rust_security
    check_dependencies_security
    check_configuration_security
    
    print_status "Security check completed!"
}

# Fix security issues
fix_security_issues() {
    print_status "Fixing security issues..."
    
    # Fix file permissions
    chmod 644 config.toml .env docker-compose.yml 2>/dev/null || true
    chmod 755 scripts/*.sh 2>/dev/null || true
    
    # Remove backup files
    find . -name "*.backup" -o -name "*.bak" -o -name "*~" -delete 2>/dev/null || true
    
    # Update dependencies
    if command -v cargo &> /dev/null; then
        cargo update
    fi
    
    print_success "Security issues fixed"
}

# Generate security report
generate_report() {
    local report_file="security-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating security report: $report_file"
    
    {
        echo "Hauptbuch Security Report"
        echo "========================"
        echo "Generated: $(date)"
        echo "Audit Level: $AUDIT_LEVEL"
        echo "Total Checks: $TOTAL_CHECKS"
        echo "Passed: $PASSED_CHECKS"
        echo "Failed: $FAILED_CHECKS"
        echo "Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
        echo ""
        
        echo "Detailed Results:"
        echo "================"
        for result in "${SECURITY_RESULTS[@]}"; do
            IFS=':' read -r name status message severity <<< "$result"
            echo "$name: $status - $message (Severity: $severity)"
        done
        
    } > "$report_file"
    
    print_success "Security report generated: $report_file"
}

# Show security summary
show_summary() {
    print_status "Security Summary:"
    echo "  Total Checks: $TOTAL_CHECKS"
    echo "  Passed: $PASSED_CHECKS"
    echo "  Failed: $FAILED_CHECKS"
    echo "  Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
    echo ""
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        print_success "✅ All security checks passed!"
    else
        print_error "❌ $FAILED_CHECKS security checks failed!"
    fi
}

# Main security function
main() {
    print_status "🔒 Starting Hauptbuch security check..."
    
    case $SECURITY_TYPE in
        "files")
            check_file_permissions
            ;;
        "data")
            check_sensitive_data
            ;;
        "network")
            check_network_security
            ;;
        "docker")
            check_docker_security
            ;;
        "rust")
            check_rust_security
            ;;
        "dependencies")
            check_dependencies_security
            ;;
        "config")
            check_configuration_security
            ;;
        "all")
            check_all_security
            ;;
        *)
            print_error "Unknown security type: $SECURITY_TYPE"
            echo "Valid types: files, data, network, docker, rust, dependencies, config, all"
            exit 1
            ;;
    esac
    
    if [ "$FIX_ISSUES" = "true" ]; then
        fix_security_issues
    fi
    
    show_summary
    
    if [ "$AUDIT_LEVEL" = "high" ]; then
        generate_report
    fi
    
    # Exit with error code if any security checks failed
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        exit 1
    fi
}

# Handle command line arguments
case "${1:-help}" in
    "files"|"data"|"network"|"docker"|"rust"|"dependencies"|"config"|"all")
        main "$@"
        ;;
    "fix")
        fix_security_issues
        ;;
    "report")
        generate_report
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Security Script"
        echo ""
        echo "Usage: $0 [SECURITY_TYPE] [AUDIT_LEVEL] [FIX_ISSUES]"
        echo ""
        echo "Security Types:"
        echo "  files        - File permissions and security"
        echo "  data         - Sensitive data exposure"
        echo "  network      - Network security"
        echo "  docker       - Docker security"
        echo "  rust         - Rust code security"
        echo "  dependencies - Dependency security"
        echo "  config       - Configuration security"
        echo "  all          - All security checks (default)"
        echo ""
        echo "Commands:"
        echo "  fix          - Fix security issues"
        echo "  report       - Generate security report"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  AUDIT_LEVEL  - Audit level: low, medium, high (default: medium)"
        echo "  FIX_ISSUES   - Fix issues automatically (default: false)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
