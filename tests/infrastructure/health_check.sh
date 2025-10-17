#!/bin/bash

# Hauptbuch Network Health Check Script
# This script checks the health of the Hauptbuch blockchain network

set -e

# Configuration
RPC_URL="http://localhost:8080"
WS_URL="ws://localhost:8081"
TIMEOUT=10
MAX_RETRIES=3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi
}

# Check if jq is available
check_jq() {
    if ! command -v jq &> /dev/null; then
        log_warn "jq is not installed, JSON parsing will be limited"
    fi
}

# Test RPC endpoint
test_rpc_endpoint() {
    local endpoint=$1
    local method=$2
    local expected_field=$3
    
    log_info "Testing RPC endpoint: $method"
    
    local response
    local status_code
    
    response=$(curl -s -w "\n%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"$method\",\"params\":{},\"id\":1}" \
        --connect-timeout $TIMEOUT \
        "$endpoint" 2>/dev/null || echo -e "\n000")
    
    status_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "200" ]; then
        if command -v jq &> /dev/null; then
            if echo "$response_body" | jq -e ".$expected_field" > /dev/null 2>&1; then
                log_info "✅ $method endpoint is healthy"
                return 0
            else
                log_error "❌ $method endpoint returned invalid response"
                return 1
            fi
        else
            log_info "✅ $method endpoint responded (status: $status_code)"
            return 0
        fi
    else
        log_error "❌ $method endpoint failed (status: $status_code)"
        return 1
    fi
}

# Test network connectivity
test_network_connectivity() {
    log_info "Testing network connectivity..."
    
    # Test RPC port
    if curl -s --connect-timeout $TIMEOUT "$RPC_URL" > /dev/null 2>&1; then
        log_info "✅ RPC port is accessible"
    else
        log_error "❌ RPC port is not accessible"
        return 1
    fi
    
    # Test WebSocket port (basic connectivity)
    if timeout $TIMEOUT bash -c "echo > /dev/tcp/localhost/8081" 2>/dev/null; then
        log_info "✅ WebSocket port is accessible"
    else
        log_warn "⚠️ WebSocket port is not accessible"
    fi
}

# Test core RPC methods
test_core_rpc_methods() {
    log_info "Testing core RPC methods..."
    
    local failed_tests=0
    
    # Test network info
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getNetworkInfo" "result"; then
        ((failed_tests++))
    fi
    
    # Test node status
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getNodeStatus" "result"; then
        ((failed_tests++))
    fi
    
    # Test chain info
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getChainInfo" "result"; then
        ((failed_tests++))
    fi
    
    # Test validator set
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getValidatorSet" "result"; then
        ((failed_tests++))
    fi
    
    # Test peer list
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getPeerList" "result"; then
        ((failed_tests++))
    fi
    
    if [ $failed_tests -eq 0 ]; then
        log_info "✅ All core RPC methods are healthy"
        return 0
    else
        log_error "❌ $failed_tests core RPC methods failed"
        return 1
    fi
}

# Test crypto RPC methods
test_crypto_rpc_methods() {
    log_info "Testing crypto RPC methods..."
    
    local failed_tests=0
    
    # Test keypair generation
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_generateKeypair" "result"; then
        ((failed_tests++))
    fi
    
    # Test signature verification
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_verifySignature" "result"; then
        ((failed_tests++))
    fi
    
    if [ $failed_tests -eq 0 ]; then
        log_info "✅ All crypto RPC methods are healthy"
        return 0
    else
        log_error "❌ $failed_tests crypto RPC methods failed"
        return 1
    fi
}

# Test transaction RPC methods
test_transaction_rpc_methods() {
    log_info "Testing transaction RPC methods..."
    
    local failed_tests=0
    
    # Test balance query
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getBalance" "result"; then
        ((failed_tests++))
    fi
    
    # Test nonce query
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getNonce" "result"; then
        ((failed_tests++))
    fi
    
    # Test transaction status
    if ! test_rpc_endpoint "$RPC_URL" "hauptbuch_getTransactionStatus" "result"; then
        ((failed_tests++))
    fi
    
    if [ $failed_tests -eq 0 ]; then
        log_info "✅ All transaction RPC methods are healthy"
        return 0
    else
        log_error "❌ $failed_tests transaction RPC methods failed"
        return 1
    fi
}

# Test system resources
test_system_resources() {
    log_info "Testing system resources..."
    
    # Check available memory
    local available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$available_memory" -gt 100 ]; then
        log_info "✅ Sufficient memory available (${available_memory}MB)"
    else
        log_warn "⚠️ Low memory available (${available_memory}MB)"
    fi
    
    # Check disk space
    local available_disk=$(df -h / | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "$available_disk" -gt 1 ]; then
        log_info "✅ Sufficient disk space available (${available_disk}GB)"
    else
        log_warn "⚠️ Low disk space available (${available_disk}GB)"
    fi
}

# Test process health
test_process_health() {
    log_info "Testing process health..."
    
    # Check if hauptbuch process is running
    if pgrep -f "hauptbuch" > /dev/null; then
        log_info "✅ Hauptbuch process is running"
        
        # Get process info
        local pid=$(pgrep -f "hauptbuch" | head -n1)
        local cpu_usage=$(ps -p "$pid" -o %cpu --no-headers | tr -d ' ')
        local memory_usage=$(ps -p "$pid" -o %mem --no-headers | tr -d ' ')
        
        log_info "Process info: PID=$pid, CPU=${cpu_usage}%, Memory=${memory_usage}%"
        
        # Check if CPU usage is reasonable
        if (( $(echo "$cpu_usage < 100" | bc -l) )); then
            log_info "✅ CPU usage is normal"
        else
            log_warn "⚠️ High CPU usage detected"
        fi
        
        # Check if memory usage is reasonable
        if (( $(echo "$memory_usage < 50" | bc -l) )); then
            log_info "✅ Memory usage is normal"
        else
            log_warn "⚠️ High memory usage detected"
        fi
    else
        log_error "❌ Hauptbuch process is not running"
        return 1
    fi
}

# Main health check function
main() {
    log_info "🔍 Starting Hauptbuch network health check..."
    
    local overall_status=0
    
    # Check prerequisites
    check_curl
    check_jq
    
    # Run health checks
    if ! test_network_connectivity; then
        overall_status=1
    fi
    
    if ! test_process_health; then
        overall_status=1
    fi
    
    if ! test_core_rpc_methods; then
        overall_status=1
    fi
    
    if ! test_crypto_rpc_methods; then
        overall_status=1
    fi
    
    if ! test_transaction_rpc_methods; then
        overall_status=1
    fi
    
    test_system_resources
    
    # Final status
    if [ $overall_status -eq 0 ]; then
        log_info "🎉 All health checks passed! Network is healthy."
        exit 0
    else
        log_error "💥 Some health checks failed! Network may have issues."
        exit 1
    fi
}

# Run main function
main "$@"

