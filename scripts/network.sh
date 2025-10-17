#!/bin/bash

# Hauptbuch Network Script
# This script manages and monitors Hauptbuch network configuration

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
NETWORK_TYPE=${1:-all}
INTERFACE=${2:-eth0}
PORT=${3:-8545}
TIMEOUT=${4:-5}

# Network results
NETWORK_RESULTS=()
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Add network result
add_result() {
    local check_name=$1
    local status=$2
    local message=$3
    local value=$4
    
    NETWORK_RESULTS+=("$check_name:$status:$message:$value")
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$status" = "PASS" ]; then
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
        print_success "$check_name: $message ($value)"
    elif [ "$status" = "FAIL" ]; then
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
        print_error "$check_name: $message ($value)"
    else
        print_warning "$check_name: $message ($value)"
    fi
}

# Check network connectivity
check_network_connectivity() {
    print_status "Checking network connectivity..."
    
    # Check internet connectivity
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        add_result "Internet Connectivity" "PASS" "Internet is accessible" "OK"
    else
        add_result "Internet Connectivity" "FAIL" "Internet is not accessible" "N/A"
    fi
    
    # Check DNS resolution
    if nslookup google.com > /dev/null 2>&1; then
        add_result "DNS Resolution" "PASS" "DNS resolution is working" "OK"
    else
        add_result "DNS Resolution" "FAIL" "DNS resolution failed" "N/A"
    fi
    
    # Check network latency
    local latency=$(ping -c 1 8.8.8.8 2>/dev/null | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}' || echo "0")
    if (( $(echo "$latency < 100" | bc -l) )); then
        add_result "Network Latency" "PASS" "Network latency is good" "${latency}ms"
    else
        add_result "Network Latency" "FAIL" "Network latency is high" "${latency}ms"
    fi
}

# Check network interfaces
check_network_interfaces() {
    print_status "Checking network interfaces..."
    
    # Check interface status
    if ip link show $INTERFACE > /dev/null 2>&1; then
        add_result "Interface Status" "PASS" "Interface $INTERFACE is available" "OK"
    else
        add_result "Interface Status" "FAIL" "Interface $INTERFACE not found" "N/A"
    fi
    
    # Check interface IP
    local interface_ip=$(ip addr show $INTERFACE 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 | head -1)
    if [ -n "$interface_ip" ]; then
        add_result "Interface IP" "PASS" "Interface has IP address" "$interface_ip"
    else
        add_result "Interface IP" "FAIL" "Interface has no IP address" "N/A"
    fi
    
    # Check interface status
    local interface_status=$(ip link show $INTERFACE 2>/dev/null | grep -o "state [A-Z]*" | cut -d' ' -f2)
    if [ "$interface_status" = "UP" ]; then
        add_result "Interface State" "PASS" "Interface is up" "$interface_status"
    else
        add_result "Interface State" "FAIL" "Interface is down" "$interface_status"
    fi
}

# Check network ports
check_network_ports() {
    print_status "Checking network ports..."
    
    # Check if port is listening
    if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
        add_result "Port Status" "PASS" "Port $PORT is listening" "OK"
    else
        add_result "Port Status" "FAIL" "Port $PORT is not listening" "N/A"
    fi
    
    # Check port accessibility
    if timeout $TIMEOUT bash -c "</dev/tcp/localhost/$PORT" 2>/dev/null; then
        add_result "Port Accessibility" "PASS" "Port $PORT is accessible" "OK"
    else
        add_result "Port Accessibility" "FAIL" "Port $PORT is not accessible" "N/A"
    fi
    
    # Check port security
    local open_ports=$(netstat -tuln 2>/dev/null | grep LISTEN | wc -l)
    if [ "$open_ports" -le 20 ]; then
        add_result "Port Security" "PASS" "Reasonable number of open ports" "$open_ports"
    else
        add_result "Port Security" "FAIL" "Too many open ports" "$open_ports"
    fi
}

# Check network security
check_network_security() {
    print_status "Checking network security..."
    
    # Check firewall status
    if command -v ufw &> /dev/null; then
        local firewall_status=$(ufw status | grep "Status:" | cut -d: -f2 | xargs)
        if [ "$firewall_status" = "active" ]; then
            add_result "Firewall Status" "PASS" "Firewall is active" "$firewall_status"
        else
            add_result "Firewall Status" "FAIL" "Firewall is not active" "$firewall_status"
        fi
    else
        add_result "Firewall Status" "WARN" "Firewall not installed" "N/A"
    fi
    
    # Check for exposed services
    local exposed_services=$(netstat -tuln 2>/dev/null | grep -E ":(22|23|21|25|53|80|443|993|995)" | wc -l)
    if [ "$exposed_services" -eq 0 ]; then
        add_result "Exposed Services" "PASS" "No critical services exposed" "$exposed_services"
    else
        add_result "Exposed Services" "FAIL" "Critical services exposed" "$exposed_services"
    fi
    
    # Check for suspicious connections
    local suspicious_connections=$(netstat -tuln 2>/dev/null | grep -E ":(6666|6667|6668|6669|1337)" | wc -l)
    if [ "$suspicious_connections" -eq 0 ]; then
        add_result "Suspicious Connections" "PASS" "No suspicious connections" "$suspicious_connections"
    else
        add_result "Suspicious Connections" "FAIL" "Suspicious connections found" "$suspicious_connections"
    fi
}

# Check network performance
check_network_performance() {
    print_status "Checking network performance..."
    
    # Check bandwidth
    if command -v iperf3 &> /dev/null; then
        local bandwidth=$(iperf3 -c 8.8.8.8 -t 1 2>/dev/null | grep "sender" | awk '{print $7}' || echo "0")
        if [ "$bandwidth" != "0" ]; then
            add_result "Network Bandwidth" "PASS" "Network bandwidth is available" "${bandwidth}Mbps"
        else
            add_result "Network Bandwidth" "FAIL" "Network bandwidth test failed" "N/A"
        fi
    else
        add_result "Network Bandwidth" "WARN" "Bandwidth test tool not available" "N/A"
    fi
    
    # Check packet loss
    local packet_loss=$(ping -c 10 8.8.8.8 2>/dev/null | grep "packet loss" | awk '{print $6}' | cut -d% -f1 || echo "0")
    if [ "$packet_loss" -eq 0 ]; then
        add_result "Packet Loss" "PASS" "No packet loss" "${packet_loss}%"
    else
        add_result "Packet Loss" "FAIL" "Packet loss detected" "${packet_loss}%"
    fi
    
    # Check network stability
    local network_stability=$(ping -c 5 8.8.8.8 2>/dev/null | grep "time=" | wc -l)
    if [ "$network_stability" -eq 5 ]; then
        add_result "Network Stability" "PASS" "Network is stable" "$network_stability/5"
    else
        add_result "Network Stability" "FAIL" "Network instability detected" "$network_stability/5"
    fi
}

# Check all network
check_all_network() {
    print_status "🌐 Starting comprehensive network analysis..."
    
    check_network_connectivity
    check_network_interfaces
    check_network_ports
    check_network_security
    check_network_performance
    
    print_status "Network analysis completed!"
}

# Configure network
configure_network() {
    print_status "Configuring network..."
    
    # Configure interface
    if [ -n "$INTERFACE" ]; then
        print_status "Configuring interface: $INTERFACE"
        
        # Bring interface up
        sudo ip link set $INTERFACE up 2>/dev/null || true
        
        # Configure IP if needed
        if [ -n "$INTERFACE_IP" ]; then
            sudo ip addr add $INTERFACE_IP dev $INTERFACE 2>/dev/null || true
        fi
        
        print_success "Interface configured"
    fi
    
    # Configure firewall
    if command -v ufw &> /dev/null; then
        print_status "Configuring firewall..."
        
        # Allow Hauptbuch ports
        sudo ufw allow 30303/tcp 2>/dev/null || true
        sudo ufw allow 8545/tcp 2>/dev/null || true
        sudo ufw allow 8546/tcp 2>/dev/null || true
        
        print_success "Firewall configured"
    fi
    
    # Configure DNS
    if [ -n "$DNS_SERVER" ]; then
        print_status "Configuring DNS server: $DNS_SERVER"
        echo "nameserver $DNS_SERVER" | sudo tee /etc/resolv.conf > /dev/null
        print_success "DNS configured"
    fi
}

# Monitor network
monitor_network() {
    print_status "Monitoring network..."
    print_status "Monitoring interval: 5s"
    print_status "Press Ctrl+C to stop"
    echo ""
    
    while true; do
        print_status "Network status: $(date)"
        
        # Check connectivity
        if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
            print_success "Internet: OK"
        else
            print_error "Internet: FAILED"
        fi
        
        # Check latency
        local latency=$(ping -c 1 8.8.8.8 2>/dev/null | grep "time=" | awk -F'time=' '{print $2}' | awk '{print $1}' || echo "0")
        print_status "Latency: ${latency}ms"
        
        # Check port status
        if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
            print_success "Port $PORT: OK"
        else
            print_error "Port $PORT: FAILED"
        fi
        
        echo ""
        sleep 5
    done
}

# Generate network report
generate_report() {
    local report_file="network-report-$(date +%Y%m%d-%H%M%S).txt"
    
    print_status "Generating network report: $report_file"
    
    {
        echo "Hauptbuch Network Report"
        echo "======================="
        echo "Generated: $(date)"
        echo "Interface: $INTERFACE"
        echo "Port: $PORT"
        echo "Timeout: $TIMEOUT"
        echo "Total Checks: $TOTAL_CHECKS"
        echo "Passed: $PASSED_CHECKS"
        echo "Failed: $FAILED_CHECKS"
        echo "Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
        echo ""
        
        echo "Network Configuration:"
        echo "  Interface: $INTERFACE"
        echo "  IP Address: $(ip addr show $INTERFACE 2>/dev/null | grep "inet " | awk '{print $2}' | cut -d/ -f1 | head -1)"
        echo "  Gateway: $(ip route | grep default | awk '{print $3}')"
        echo "  DNS: $(cat /etc/resolv.conf | grep nameserver | head -1 | awk '{print $2}')"
        echo ""
        
        echo "Detailed Results:"
        echo "================"
        for result in "${NETWORK_RESULTS[@]}"; do
            IFS=':' read -r name status message value <<< "$result"
            echo "$name: $status - $message ($value)"
        done
        
    } > "$report_file"
    
    print_success "Network report generated: $report_file"
}

# Show network summary
show_summary() {
    print_status "Network Summary:"
    echo "  Total Checks: $TOTAL_CHECKS"
    echo "  Passed: $PASSED_CHECKS"
    echo "  Failed: $FAILED_CHECKS"
    echo "  Warnings: $((TOTAL_CHECKS - PASSED_CHECKS - FAILED_CHECKS))"
    echo ""
    
    if [ "$FAILED_CHECKS" -eq 0 ]; then
        print_success "✅ All network checks passed!"
    else
        print_error "❌ $FAILED_CHECKS network checks failed!"
    fi
}

# Main network function
main() {
    print_status "🌐 Starting Hauptbuch network analysis..."
    
    case $NETWORK_TYPE in
        "connectivity")
            check_network_connectivity
            ;;
        "interfaces")
            check_network_interfaces
            ;;
        "ports")
            check_network_ports
            ;;
        "security")
            check_network_security
            ;;
        "performance")
            check_network_performance
            ;;
        "all")
            check_all_network
            ;;
        *)
            print_error "Unknown network type: $NETWORK_TYPE"
            echo "Valid types: connectivity, interfaces, ports, security, performance, all"
            exit 1
            ;;
    esac
    
    show_summary
}

# Handle command line arguments
case "${1:-help}" in
    "connectivity"|"interfaces"|"ports"|"security"|"performance"|"all")
        main "$@"
        ;;
    "configure")
        configure_network
        ;;
    "monitor")
        monitor_network
        ;;
    "report")
        generate_report
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Network Script"
        echo ""
        echo "Usage: $0 [NETWORK_TYPE] [INTERFACE] [PORT] [TIMEOUT]"
        echo ""
        echo "Network Types:"
        echo "  connectivity - Network connectivity"
        echo "  interfaces   - Network interfaces"
        echo "  ports        - Network ports"
        echo "  security     - Network security"
        echo "  performance  - Network performance"
        echo "  all          - All network checks (default)"
        echo ""
        echo "Commands:"
        echo "  configure    - Configure network"
        echo "  monitor      - Monitor network"
        echo "  report       - Generate network report"
        echo "  help         - Show this help message"
        echo ""
        echo "Parameters:"
        echo "  INTERFACE    - Network interface (default: eth0)"
        echo "  PORT         - Port to check (default: 8545)"
        echo "  TIMEOUT      - Timeout in seconds (default: 5)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
