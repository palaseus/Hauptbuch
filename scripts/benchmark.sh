#!/bin/bash

# Hauptbuch Benchmarking Script
# This script runs comprehensive benchmarks for Hauptbuch

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
BENCHMARK_TYPE=${1:-all}
DURATION=${2:-60}
THREADS=${3:-4}
RESULTS_DIR="benchmarks/$(date +%Y%m%d-%H%M%S)"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run consensus benchmarks
benchmark_consensus() {
    print_status "Running consensus benchmarks..."
    
    local output_file="$RESULTS_DIR/consensus-benchmarks.txt"
    
    {
        echo "Consensus Benchmarks"
        echo "==================="
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run PoS consensus benchmarks
        cargo bench --bench pos_benchmarks -- --nocapture
        
    } > "$output_file"
    
    print_success "Consensus benchmarks completed: $output_file"
}

# Run cryptography benchmarks
benchmark_cryptography() {
    print_status "Running cryptography benchmarks..."
    
    local output_file="$RESULTS_DIR/crypto-benchmarks.txt"
    
    {
        echo "Cryptography Benchmarks"
        echo "======================="
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run NIST PQC benchmarks
        cargo bench --bench nist_pqc_benchmarks -- --nocapture
        
        # Run classical crypto benchmarks
        cargo bench --bench classical_crypto_benchmarks -- --nocapture
        
        # Run zk-proof benchmarks
        cargo bench --bench zk_proof_benchmarks -- --nocapture
        
    } > "$output_file"
    
    print_success "Cryptography benchmarks completed: $output_file"
}

# Run performance benchmarks
benchmark_performance() {
    print_status "Running performance benchmarks..."
    
    local output_file="$RESULTS_DIR/performance-benchmarks.txt"
    
    {
        echo "Performance Benchmarks"
        echo "======================"
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run Block-STM benchmarks
        cargo bench --bench block_stm_benchmarks -- --nocapture
        
        # Run optimistic validation benchmarks
        cargo bench --bench optimistic_validation_benchmarks -- --nocapture
        
        # Run QUIC networking benchmarks
        cargo bench --bench quic_networking_benchmarks -- --nocapture
        
        # Run Sealevel parallel benchmarks
        cargo bench --bench sealevel_parallel_benchmarks -- --nocapture
        
        # Run state cache benchmarks
        cargo bench --bench state_cache_benchmarks -- --nocapture
        
    } > "$output_file"
    
    print_success "Performance benchmarks completed: $output_file"
}

# Run network benchmarks
benchmark_network() {
    print_status "Running network benchmarks..."
    
    local output_file="$RESULTS_DIR/network-benchmarks.txt"
    
    {
        echo "Network Benchmarks"
        echo "=================="
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run P2P benchmarks
        cargo bench --bench p2p_benchmarks -- --nocapture
        
        # Run cross-chain benchmarks
        cargo bench --bench cross_chain_benchmarks -- --nocapture
        
    } > "$output_file"
    
    print_success "Network benchmarks completed: $output_file"
}

# Run L2 benchmarks
benchmark_l2() {
    print_status "Running L2 benchmarks..."
    
    local output_file="$RESULTS_DIR/l2-benchmarks.txt"
    
    {
        echo "L2 Benchmarks"
        echo "============="
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run rollup benchmarks
        cargo bench --bench rollup_benchmarks -- --nocapture
        
        # Run zkEVM benchmarks
        cargo bench --bench zkevm_benchmarks -- --nocapture
        
        # Run SP1 zkVM benchmarks
        cargo bench --bench sp1_zkvm_benchmarks -- --nocapture
        
        # Run EIP-4844 benchmarks
        cargo bench --bench eip4844_benchmarks -- --nocapture
        
    } > "$output_file"
    
    print_success "L2 benchmarks completed: $output_file"
}

# Run security benchmarks
benchmark_security() {
    print_status "Running security benchmarks..."
    
    local output_file="$RESULTS_DIR/security-benchmarks.txt"
    
    {
        echo "Security Benchmarks"
        echo "==================="
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run audit benchmarks
        cargo bench --bench audit_benchmarks -- --nocapture
        
        # Run formal verification benchmarks
        cargo bench --bench formal_verification_benchmarks -- --nocapture
        
        # Run MEV protection benchmarks
        cargo bench --bench mev_protection_benchmarks -- --nocapture
        
    } > "$output_file"
    
    print_success "Security benchmarks completed: $output_file"
}

# Run stress tests
run_stress_tests() {
    print_status "Running stress tests..."
    
    local output_file="$RESULTS_DIR/stress-tests.txt"
    
    {
        echo "Stress Tests"
        echo "============"
        echo "Timestamp: $(date)"
        echo "Duration: ${DURATION}s"
        echo "Threads: $THREADS"
        echo ""
        
        # Run high-load tests
        cargo test --test stress_tests -- --nocapture
        
        # Run chaos engineering tests
        cargo test --test chaos_engineering -- --nocapture
        
    } > "$output_file"
    
    print_success "Stress tests completed: $output_file"
}

# Generate benchmark report
generate_report() {
    print_status "Generating benchmark report..."
    
    local report_file="$RESULTS_DIR/benchmark-report.html"
    
    {
        echo "<!DOCTYPE html>"
        echo "<html>"
        echo "<head>"
        echo "  <title>Hauptbuch Benchmark Report</title>"
        echo "  <style>"
        echo "    body { font-family: Arial, sans-serif; margin: 20px; }"
        echo "    h1, h2 { color: #333; }"
        echo "    .success { color: green; }"
        echo "    .warning { color: orange; }"
        echo "    .error { color: red; }"
        echo "    pre { background: #f5f5f5; padding: 10px; border-radius: 5px; }"
        echo "  </style>"
        echo "</head>"
        echo "<body>"
        echo "  <h1>Hauptbuch Benchmark Report</h1>"
        echo "  <p><strong>Generated:</strong> $(date)</p>"
        echo "  <p><strong>Environment:</strong> $(uname -a)</p>"
        echo "  <p><strong>Duration:</strong> ${DURATION}s</p>"
        echo "  <p><strong>Threads:</strong> $THREADS</p>"
        echo ""
        echo "  <h2>System Information</h2>"
        echo "  <pre>"
        echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
        echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
        echo "Disk: $(df -h / | awk 'NR==2{print $2}')"
        echo "  </pre>"
        echo ""
        echo "  <h2>Benchmark Results</h2>"
        
        # Include all benchmark results
        for file in "$RESULTS_DIR"/*.txt; do
            if [ -f "$file" ]; then
                echo "  <h3>$(basename "$file" .txt)</h3>"
                echo "  <pre>"
                cat "$file"
                echo "  </pre>"
            fi
        done
        
        echo "</body>"
        echo "</html>"
        
    } > "$report_file"
    
    print_success "Benchmark report generated: $report_file"
}

# Run all benchmarks
run_all_benchmarks() {
    print_status "Running all benchmarks..."
    
    benchmark_consensus
    benchmark_cryptography
    benchmark_performance
    benchmark_network
    benchmark_l2
    benchmark_security
    run_stress_tests
    
    generate_report
    
    print_success "All benchmarks completed!"
}

# Main benchmarking function
main() {
    print_status "🚀 Starting Hauptbuch benchmarks..."
    
    case $BENCHMARK_TYPE in
        "consensus")
            benchmark_consensus
            ;;
        "crypto")
            benchmark_cryptography
            ;;
        "performance")
            benchmark_performance
            ;;
        "network")
            benchmark_network
            ;;
        "l2")
            benchmark_l2
            ;;
        "security")
            benchmark_security
            ;;
        "stress")
            run_stress_tests
            ;;
        "all")
            run_all_benchmarks
            ;;
        *)
            print_error "Unknown benchmark type: $BENCHMARK_TYPE"
            echo "Valid types: consensus, crypto, performance, network, l2, security, stress, all"
            exit 1
            ;;
    esac
    
    print_success "✅ Benchmarks completed successfully!"
    print_status "Results saved to: $RESULTS_DIR"
}

# Handle command line arguments
case "${1:-help}" in
    "consensus"|"crypto"|"performance"|"network"|"l2"|"security"|"stress"|"all")
        main "$@"
        ;;
    "help"|"--help"|"-h")
        echo "Hauptbuch Benchmarking Script"
        echo ""
        echo "Usage: $0 [BENCHMARK_TYPE] [DURATION] [THREADS]"
        echo ""
        echo "Benchmark Types:"
        echo "  consensus   - Consensus algorithm benchmarks"
        echo "  crypto      - Cryptography benchmarks"
        echo "  performance - Performance optimization benchmarks"
        echo "  network     - Network and P2P benchmarks"
        echo "  l2          - Layer 2 solution benchmarks"
        echo "  security    - Security and audit benchmarks"
        echo "  stress      - Stress testing"
        echo "  all         - Run all benchmarks (default)"
        echo ""
        echo "Parameters:"
        echo "  DURATION    - Benchmark duration in seconds (default: 60)"
        echo "  THREADS     - Number of threads to use (default: 4)"
        ;;
    *)
        print_error "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac
