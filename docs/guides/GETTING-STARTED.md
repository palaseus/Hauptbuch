# Getting Started with Hauptbuch

## Overview

Welcome to Hauptbuch, a quantum-resistant blockchain platform designed for the future of decentralized applications. This guide will help you get started with the platform, from installation to running your first transactions.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running Your First Node](#running-your-first-node)
- [Creating Your First Account](#creating-your-first-account)
- [Making Your First Transaction](#making-your-first-transaction)
- [Exploring the Network](#exploring-the-network)
- [Next Steps](#next-steps)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Memory**: Minimum 8GB RAM, 16GB recommended
- **Storage**: At least 100GB free disk space
- **Network**: Stable internet connection

### Software Dependencies

- **Rust**: Version 1.70 or later
- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later
- **Git**: For cloning the repository

### Hardware Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: SSD recommended for better performance
- **Network**: Stable internet connection

## Installation

### Option 1: Quick Start with Docker

The easiest way to get started is using Docker:

```bash
# Clone the repository
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch

# Run the setup script
./scripts/setup.sh

# Start the development environment
./scripts/dev.sh
```

### Option 2: Manual Installation

For development or production use:

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install system dependencies
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev cmake libclang-dev

# Clone and build
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch
cargo build --release
```

### Option 3: Using the Installer Script

```bash
# Download and run the installer
curl -sSL https://install.hauptbuch.org | bash

# Follow the interactive prompts
```

## Configuration

### Basic Configuration

Create a configuration file:

```bash
# Copy the example configuration
cp config.toml.example config.toml

# Edit the configuration
nano config.toml
```

### Key Configuration Options

```toml
# Core Configuration
[core]
network_id = "hauptbuch-testnet-1"
chain_id = 1337
log_level = "info"
data_dir = "/var/lib/hauptbuch"

# Consensus Configuration
[consensus]
validator_set_size = 100
block_time_ms = 5000
epoch_length_blocks = 1000
vdf_difficulty = 1000000
pow_difficulty = "0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
slashing_enabled = true
slashing_threshold = 0.05

# Network Configuration
[network]
listen_address = "0.0.0.0:8080"
bootnodes = ["/ip4/127.0.0.1/tcp/8080/p2p/Qm..."]
max_connections = 100
enable_quic = true

# Cryptography Configuration
[crypto]
default_signature_scheme = "ml-dsa"
default_key_exchange_scheme = "ml-kem"
hybrid_mode_enabled = true
classical_signature_fallback = "p256"
classical_key_exchange_fallback = "x25519"
```

### Environment Variables

Set up environment variables:

```bash
# Copy the example environment file
cp env.example .env

# Edit the environment variables
nano .env
```

Key environment variables:

```bash
# Core Configuration
HAUPTBUCH_NETWORK_ID="hauptbuch-testnet-1"
HAUPTBUCH_CHAIN_ID=1337
HAUPTBUCH_LOG_LEVEL="info"
HAUPTBUCH_DATA_DIR="/var/lib/hauptbuch"

# Consensus Configuration
HAUPTBUCH_VALIDATOR_SET_SIZE=100
HAUPTBUCH_BLOCK_TIME_MS=5000
HAUPTBUCH_EPOCH_LENGTH_BLOCKS=1000
HAUPTBUCH_VDF_DIFFICULTY=1000000
HAUPTBUCH_POW_DIFFICULTY="0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
HAUPTBUCH_SLASHING_ENABLED=true
HAUPTBUCH_SLASHING_THRESHOLD=0.05

# Network Configuration
HAUPTBUCH_LISTEN_ADDRESS="0.0.0.0:8080"
HAUPTBUCH_BOOTNODES="/ip4/127.0.0.1/tcp/8080/p2p/Qm..."
HAUPTBUCH_MAX_CONNECTIONS=100
HAUPTBUCH_ENABLE_QUIC=true

# Cryptography Configuration
HAUPTBUCH_DEFAULT_SIGNATURE_SCHEME="ml-dsa"
HAUPTBUCH_DEFAULT_KEY_EXCHANGE_SCHEME="ml-kem"
HAUPTBUCH_HYBRID_MODE_ENABLED=true
HAUPTBUCH_CLASSICAL_SIGNATURE_FALLBACK="p256"
HAUPTBUCH_CLASSICAL_KEY_EXCHANGE_FALLBACK="x25519"
```

## Running Your First Node

### Start the Node

```bash
# Start the Hauptbuch node
./target/release/hauptbuch

# Or using Docker
docker-compose up -d
```

### Verify the Node is Running

```bash
# Check node status
curl http://localhost:8080/status

# Check node logs
docker-compose logs -f hauptbuch-node
```

### Connect to the Network

```bash
# Connect to the testnet
./target/release/hauptbuch --network testnet

# Connect to the mainnet
./target/release/hauptbuch --network mainnet
```

## Creating Your First Account

### Using the CLI

```bash
# Create a new account
./target/release/hauptbuch account create

# List your accounts
./target/release/hauptbuch account list

# Get account balance
./target/release/hauptbuch account balance <address>
```

### Using the SDK

```rust
use hauptbuch_sdk::{Client, Account, QuantumResistantCrypto};

// Create a new account
let account = Account::new()?;
let address = account.address();
let private_key = account.private_key();

// Create a client
let client = Client::new("http://localhost:8080")?;

// Get account balance
let balance = client.get_balance(&address).await?;
println!("Account balance: {}", balance);
```

### Using the Web Interface

1. Open your browser and navigate to `http://localhost:3000`
2. Click "Create Account"
3. Follow the prompts to generate your account
4. Save your private key securely

## Making Your First Transaction

### Basic Transaction

```rust
use hauptbuch_sdk::{Client, Transaction, Account};

// Create a transaction
let transaction = Transaction::new()
    .from(sender_address)
    .to(recipient_address)
    .value(1000000000000000000) // 1 HBK
    .gas_limit(21000)
    .gas_price(20000000000);

// Sign the transaction
let signed_tx = account.sign_transaction(transaction)?;

// Send the transaction
let tx_hash = client.send_transaction(&signed_tx).await?;
println!("Transaction hash: {}", tx_hash);
```

### Smart Contract Interaction

```rust
use hauptbuch_sdk::{Client, Contract, Account};

// Deploy a smart contract
let contract_code = include_bytes!("contract.wasm");
let contract = Contract::new(contract_code)?;
let contract_address = client.deploy_contract(&contract).await?;

// Call a contract function
let result = client.call_contract_function(
    &contract_address,
    "transfer",
    &[recipient_address, amount],
).await?;
```

### Cross-Chain Transaction

```rust
use hauptbuch_sdk::{Client, CrossChainTransaction, Account};

// Create a cross-chain transaction
let cross_chain_tx = CrossChainTransaction::new()
    .from(sender_address)
    .to(recipient_address)
    .value(1000000000000000000)
    .target_chain("ethereum")
    .bridge_address("0x...");

// Sign and send
let signed_tx = account.sign_transaction(cross_chain_tx)?;
let tx_hash = client.send_cross_chain_transaction(&signed_tx).await?;
```

## Exploring the Network

### Network Status

```bash
# Check network status
curl http://localhost:8080/network/status

# Check connected peers
curl http://localhost:8080/network/peers

# Check block height
curl http://localhost:8080/network/block_height
```

### Block Explorer

1. Open your browser and navigate to `http://localhost:3000/explorer`
2. Browse blocks, transactions, and accounts
3. Search for specific addresses or transaction hashes
4. View network statistics and metrics

### Monitoring Dashboard

1. Open your browser and navigate to `http://localhost:3000/dashboard`
2. View real-time network metrics
3. Monitor node performance
4. Track transaction throughput

## Next Steps

### Learn More

- **Architecture**: Read the [Architecture Overview](../architecture/OVERVIEW.md)
- **Cryptography**: Explore [Quantum-Resistant Cryptography](../modules/crypto/QUANTUM-RESISTANT.md)
- **Consensus**: Understand [Proof of Stake](../modules/consensus/POS.md)
- **Cross-Chain**: Learn about [Cross-Chain Interoperability](../modules/cross-chain/BRIDGE.md)

### Development

- **Smart Contracts**: Start building with [Smart Contract Development](../guides/SMART-CONTRACTS.md)
- **SDK**: Use the [Rust SDK](../modules/sdk/RUST-SDK.md)
- **CLI**: Master the [Command Line Interface](../modules/tools/CLI.md)

### Advanced Features

- **Governance**: Participate in [Governance](../modules/governance/PROPOSALS.md)
- **Layer 2**: Explore [Layer 2 Solutions](../modules/l2/ROLLUPS.md)
- **Account Abstraction**: Use [Account Abstraction](../modules/account-abstraction/ERC4337.md)

### Community

- **Discord**: Join our [Discord server](https://discord.gg/hauptbuch)
- **GitHub**: Contribute on [GitHub](https://github.com/hauptbuch/hauptbuch)
- **Documentation**: Read the [full documentation](../README.md)

## Troubleshooting

### Common Issues

1. **Node won't start**
   - Check if the port is already in use
   - Verify your configuration file
   - Check the logs for errors

2. **Can't connect to network**
   - Verify your network configuration
   - Check if bootnodes are accessible
   - Ensure firewall settings allow connections

3. **Transaction failed**
   - Check if you have sufficient balance
   - Verify gas settings
   - Check if the recipient address is valid

### Getting Help

- **Documentation**: Check the [documentation](../README.md)
- **Issues**: Report issues on [GitHub](https://github.com/hauptbuch/hauptbuch/issues)
- **Community**: Ask questions on [Discord](https://discord.gg/hauptbuch)

## Conclusion

Congratulations! You've successfully set up your first Hauptbuch node and made your first transaction. You're now ready to explore the full potential of the Hauptbuch blockchain platform.

For more advanced features and capabilities, continue reading the documentation and exploring the various modules and tools available in the Hauptbuch ecosystem.
