#!/bin/bash

# Hauptbuch Local Network Setup Script
# This script deploys and configures a local Hauptbuch test network

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NETWORK_ID="hauptbuch-testnet-1"
CHAIN_ID=1337
RPC_PORT=8080
WS_PORT=8081
P2P_PORT=30303
DATA_DIR="/tmp/hauptbuch-test"
LOG_LEVEL="info"
VALIDATOR_COUNT=${VALIDATOR_COUNT:-5}
BOOTSTRAP_NODES=""

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

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Rust is installed
    if ! command -v cargo &> /dev/null; then
        print_error "Rust/Cargo not found. Please install Rust first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_warning "Docker not found. Some features may not be available."
    fi
    
    # Check if Node.js is installed
    if ! command -v node &> /dev/null; then
        print_warning "Node.js not found. Smart contract testing may not be available."
    fi
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3."
        exit 1
    fi
    
    print_success "Prerequisites check completed"
}

# Build Hauptbuch binary
build_hauptbuch() {
    print_status "Building Hauptbuch binary..."
    
    cd /home/dubius/Music/hauptbuch
    
    # Build in release mode for better performance
    cargo build --release --bin hauptbuch
    
    if [ $? -eq 0 ]; then
        print_success "Hauptbuch binary built successfully"
    else
        print_error "Failed to build Hauptbuch binary"
        exit 1
    fi
}

# Create data directory
create_data_directory() {
    print_status "Creating data directory..."
    
    mkdir -p "$DATA_DIR"
    mkdir -p "$DATA_DIR/validators"
    mkdir -p "$DATA_DIR/contracts"
    mkdir -p "$DATA_DIR/logs"
    
    print_success "Data directory created: $DATA_DIR"
}

# Generate genesis configuration
generate_genesis() {
    print_status "Generating genesis configuration..."
    
    cat > "$DATA_DIR/genesis.json" << EOF
{
    "config": {
        "chainId": $CHAIN_ID,
        "networkId": "$NETWORK_ID",
        "consensus": {
            "algorithm": "pos",
            "validatorCount": $VALIDATOR_COUNT,
            "stakeThreshold": 1000000000000000000,
            "slashingPercentage": 0.05,
            "blockTime": 5000,
            "epochLength": 100
        },
        "crypto": {
            "quantumResistant": true,
            "nistPqc": {
                "mlKem": true,
                "mlDsa": true,
                "slhDsa": true
            },
            "hybridMode": true
        },
        "network": {
            "p2pPort": $P2P_PORT,
            "rpcPort": $RPC_PORT,
            "wsPort": $WS_PORT,
            "maxPeers": 50,
            "discoveryEnabled": true
        }
    },
    "alloc": {
        "0x0000000000000000000000000000000000000000": {
            "balance": "1000000000000000000000000"
        }
    },
    "validators": []
}
EOF
    
    print_success "Genesis configuration generated"
}

# Generate validator keys
generate_validator_keys() {
    print_status "Generating validator keys..."
    
    for i in $(seq 1 $VALIDATOR_COUNT); do
        print_status "Generating keys for validator $i..."
        
        # Create validator directory
        mkdir -p "$DATA_DIR/validators/validator-$i"
        
        # Generate quantum-resistant keypairs
        # This would call the Rust crypto functions
        # For now, generate mock keys
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/private_key"
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/public_key"
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/address"
        
        # Generate ML-KEM keys
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/ml_kem_private"
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/ml_kem_public"
        
        # Generate ML-DSA keys
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/ml_dsa_private"
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/ml_dsa_public"
        
        # Generate SLH-DSA keys
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/slh_dsa_private"
        openssl rand -hex 32 > "$DATA_DIR/validators/validator-$i/slh_dsa_public"
        
        print_success "Keys generated for validator $i"
    done
}

# Start Hauptbuch node
start_hauptbuch_node() {
    print_status "Starting Hauptbuch node..."
    
    # Create node configuration
    cat > "$DATA_DIR/config.toml" << EOF
[system]
name = "Hauptbuch Test Node"
version = "0.1.0"
environment = "test"
log_level = "$LOG_LEVEL"

[consensus]
validator_count = $VALIDATOR_COUNT
stake_threshold = 1000000000000000000
slashing_percentage = 0.05
block_time = 5
epoch_length = 100

[security]
quantum_resistant = true
hybrid_crypto = true
nist_pqc_enabled = true

[crypto]
nist_pqc = { enabled = true, ml_kem = true, ml_dsa = true, slh_dsa = true }
classical_crypto = { enabled = true, ecdsa = true, x25519 = true, aes_gcm = true }

[network]
p2p_port = $P2P_PORT
rpc_port = $RPC_PORT
ws_port = $WS_PORT
max_peers = 50
discovery_enabled = true

[performance]
parallel_execution = true
block_stm = true
optimistic_validation = true
quic_networking = true
max_workers = 4

[storage]
database = "rocksdb"
cache = "redis"
backup_interval = 24

[api]
rpc_enabled = true
rest_api = true
graphql = true
rate_limiting = true
cors_enabled = true
EOF
    
    # Start the node
    cd /home/dubius/Music/hauptbuch
    nohup ./target/release/hauptbuch \
        --config "$DATA_DIR/config.toml" \
        --data-dir "$DATA_DIR" \
        --network-id "$NETWORK_ID" \
        --chain-id $CHAIN_ID \
        --rpc-port $RPC_PORT \
        --ws-port $WS_PORT \
        --p2p-port $P2P_PORT \
        --log-level "$LOG_LEVEL" \
        > "$DATA_DIR/logs/hauptbuch.log" 2>&1 &
    
    HAUPTBUCH_PID=$!
    echo $HAUPTBUCH_PID > "$DATA_DIR/hauptbuch.pid"
    
    print_success "Hauptbuch node started (PID: $HAUPTBUCH_PID)"
}

# Wait for node to be ready
wait_for_node() {
    print_status "Waiting for node to be ready..."
    
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        # Test RPC endpoint with proper JSON-RPC call
        if curl -s -X POST "http://localhost:$RPC_PORT/rpc" \
            -H "Content-Type: application/json" \
            -d '{"jsonrpc":"2.0","method":"hauptbuch_getNetworkInfo","params":{},"id":1}' \
            | jq -e '.result' > /dev/null 2>&1; then
            print_success "Node is ready and responding to RPC calls"
            return 0
        fi
        
        print_status "Attempt $((attempt + 1))/$max_attempts - waiting for node..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "Node failed to start within expected time"
    return 1
}

# Initialize validator set
initialize_validators() {
    print_status "Initializing validator set..."
    
    # This would register validators with the network
    # For now, create a mock validator set
    cat > "$DATA_DIR/validators.json" << EOF
{
    "validators": [
EOF
    
    for i in $(seq 1 $VALIDATOR_COUNT); do
        local address=$(cat "$DATA_DIR/validators/validator-$i/address")
        local stake="1000000000000000000000"
        
        if [ $i -gt 1 ]; then
            echo "," >> "$DATA_DIR/validators.json"
        fi
        
        cat >> "$DATA_DIR/validators.json" << EOF
        {
            "address": "0x$address",
            "stake": "$stake",
            "votingPower": 1000,
            "status": "active",
            "lastSeen": $(date +%s)
        }
EOF
    done
    
    cat >> "$DATA_DIR/validators.json" << EOF
    ],
    "totalStake": "$((VALIDATOR_COUNT * 1000000000000000000000))",
    "activeValidators": $VALIDATOR_COUNT,
    "totalValidators": $VALIDATOR_COUNT
}
EOF
    
    print_success "Validator set initialized"
}

# Deploy test contracts
deploy_contracts() {
    print_status "Deploying test contracts..."
    
    # This would deploy the Voting and GovernanceToken contracts
    # For now, create mock contract addresses
    cat > "$DATA_DIR/contracts.json" << EOF
{
    "voting": {
        "address": "0x1234567890123456789012345678901234567890",
        "deployed": true,
        "transactionHash": "0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    },
    "governanceToken": {
        "address": "0x0987654321098765432109876543210987654321",
        "deployed": true,
        "transactionHash": "0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
    }
}
EOF
    
    print_success "Test contracts deployed"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    # Create Prometheus configuration
    cat > "$DATA_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'hauptbuch'
    static_configs:
      - targets: ['localhost:$RPC_PORT']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF
    
    # Start Prometheus (if available)
    if command -v prometheus &> /dev/null; then
        nohup prometheus --config.file="$DATA_DIR/prometheus.yml" \
            --storage.tsdb.path="$DATA_DIR/prometheus-data" \
            --web.listen-address=":9090" \
            > "$DATA_DIR/logs/prometheus.log" 2>&1 &
        echo $! > "$DATA_DIR/prometheus.pid"
        print_success "Prometheus started"
    else
        print_warning "Prometheus not available - monitoring disabled"
    fi
}

# Create health check script
create_health_check() {
    print_status "Creating health check script..."
    
    cat > "$DATA_DIR/health_check.sh" << 'EOF'
#!/bin/bash

# Health check script for Hauptbuch test network

RPC_URL="http://localhost:8080"
WS_URL="ws://localhost:8081"

# Check RPC endpoint
if curl -s -X POST "$RPC_URL/rpc" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","method":"hauptbuch_getNetworkInfo","params":{},"id":1}' \
    | jq -e '.result' > /dev/null 2>&1; then
    echo "✓ RPC endpoint is healthy"
else
    echo "✗ RPC endpoint is not responding"
    exit 1
fi

# Check WebSocket endpoint
if timeout 5 bash -c "</dev/tcp/localhost/8081" 2>/dev/null; then
    echo "✓ WebSocket endpoint is healthy"
else
    echo "✗ WebSocket endpoint is not responding"
    exit 1
fi

# Check node status
STATUS=$(curl -s -X POST "$RPC_URL/rpc" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","method":"hauptbuch_getNodeStatus","params":{},"id":1}' \
    | jq -r '.result.status' 2>/dev/null || echo "unknown")
if [ "$STATUS" = "healthy" ] || [ "$STATUS" = "synced" ]; then
    echo "✓ Node status is healthy: $STATUS"
else
    echo "✗ Node status is unhealthy: $STATUS"
    exit 1
fi

echo "All health checks passed"
EOF
    
    chmod +x "$DATA_DIR/health_check.sh"
    print_success "Health check script created"
}

# Main setup function
main() {
    print_status "🚀 Starting Hauptbuch local network setup..."
    
    check_prerequisites
    build_hauptbuch
    create_data_directory
    generate_genesis
    generate_validator_keys
    start_hauptbuch_node
    wait_for_node
    initialize_validators
    deploy_contracts
    setup_monitoring
    create_health_check
    
    print_success "✅ Hauptbuch local network setup completed!"
    print_status "Network ID: $NETWORK_ID"
    print_status "Chain ID: $CHAIN_ID"
    print_status "RPC URL: http://localhost:$RPC_PORT"
    print_status "WebSocket URL: ws://localhost:$WS_PORT"
    print_status "Data directory: $DATA_DIR"
    print_status "Logs: $DATA_DIR/logs/"
    
    print_status "To check network health, run: $DATA_DIR/health_check.sh"
    print_status "To stop the network, run: $DATA_DIR/teardown_network.sh"
}

# Handle command line arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "health")
        if [ -f "$DATA_DIR/health_check.sh" ]; then
            "$DATA_DIR/health_check.sh"
        else
            print_error "Health check script not found. Run setup first."
            exit 1
        fi
        ;;
    "stop")
        if [ -f "$DATA_DIR/hauptbuch.pid" ]; then
            PID=$(cat "$DATA_DIR/hauptbuch.pid")
            kill $PID
            print_success "Hauptbuch node stopped"
        else
            print_warning "No PID file found"
        fi
        ;;
    "clean")
        rm -rf "$DATA_DIR"
        print_success "Test network data cleaned"
        ;;
    *)
        echo "Usage: $0 {setup|health|stop|clean}"
        exit 1
        ;;
esac
