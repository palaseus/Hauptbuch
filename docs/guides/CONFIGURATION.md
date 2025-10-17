# Configuration Guide

## Overview

This guide provides comprehensive instructions for configuring Hauptbuch nodes, networks, and applications. Learn how to optimize your setup for different use cases and environments.

## Table of Contents

- [Configuration Files](#configuration-files)
- [Core Configuration](#core-configuration)
- [Network Configuration](#network-configuration)
- [Consensus Configuration](#consensus-configuration)
- [Cryptography Configuration](#cryptography-configuration)
- [Database Configuration](#database-configuration)
- [Monitoring Configuration](#monitoring-configuration)
- [Environment Variables](#environment-variables)
- [Advanced Configuration](#advanced-configuration)
- [Configuration Validation](#configuration-validation)

## Configuration Files

### Main Configuration File

The primary configuration file is `config.toml`:

```toml
# Hauptbuch Configuration File
# This file contains all configuration options for the Hauptbuch node

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

# Database Configuration
[database]
type = "rocksdb"
path = "/var/lib/hauptbuch/db"
cache_size_mb = 256
max_open_files = 1000

# Cache Configuration
[cache]
type = "redis"
address = "127.0.0.1:6379"
password = ""
ttl_seconds = 3600

# Monitoring Configuration
[monitoring]
enabled = true
metrics_address = "0.0.0.0:9090"
tracing_enabled = true
jaeger_agent_host = "127.0.0.1"
jaeger_agent_port = 6831
```

### Environment Variables

Use environment variables for sensitive or dynamic configuration:

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

# Database Configuration
HAUPTBUCH_DATABASE_TYPE="rocksdb"
HAUPTBUCH_DATABASE_PATH="/var/lib/hauptbuch/db"
HAUPTBUCH_DATABASE_CACHE_SIZE_MB=256
HAUPTBUCH_DATABASE_MAX_OPEN_FILES=1000

# Cache Configuration
HAUPTBUCH_CACHE_TYPE="redis"
HAUPTBUCH_CACHE_ADDRESS="127.0.0.1:6379"
HAUPTBUCH_CACHE_PASSWORD=""
HAUPTBUCH_CACHE_TTL_SECONDS=3600

# Monitoring Configuration
HAUPTBUCH_MONITORING_ENABLED=true
HAUPTBUCH_METRICS_ADDRESS="0.0.0.0:9090"
HAUPTBUCH_TRACING_ENABLED=true
HAUPTBUCH_JAEGER_AGENT_HOST="127.0.0.1"
HAUPTBUCH_JAEGER_AGENT_PORT=6831
```

## Core Configuration

### Basic Settings

```toml
[core]
# Network identifier
network_id = "hauptbuch-testnet-1"

# Chain identifier
chain_id = 1337

# Log level (trace, debug, info, warn, error)
log_level = "info"

# Data directory
data_dir = "/var/lib/hauptbuch"

# Enable development mode
dev_mode = false

# Enable testnet mode
testnet = true

# Enable mainnet mode
mainnet = false
```

### Advanced Core Settings

```toml
[core]
# Maximum number of concurrent operations
max_concurrent_ops = 1000

# Timeout for operations (seconds)
operation_timeout = 30

# Enable experimental features
experimental_features = false

# Enable debug mode
debug_mode = false

# Enable profiling
profiling_enabled = false

# Profiling output file
profiling_output = "/var/lib/hauptbuch/profile.json"
```

## Network Configuration

### Basic Network Settings

```toml
[network]
# Listen address
listen_address = "0.0.0.0:8080"

# Bootnodes for network discovery
bootnodes = [
    "/ip4/127.0.0.1/tcp/8080/p2p/Qm...",
    "/ip4/192.168.1.100/tcp/8080/p2p/Qm...",
]

# Maximum number of connections
max_connections = 100

# Enable QUIC protocol
enable_quic = true

# Enable TCP fallback
enable_tcp = true

# Connection timeout (seconds)
connection_timeout = 30

# Keep-alive interval (seconds)
keep_alive_interval = 60
```

### Advanced Network Settings

```toml
[network]
# Enable NAT traversal
enable_nat_traversal = true

# Enable UPnP
enable_upnp = true

# Enable hole punching
enable_hole_punching = true

# Enable relay connections
enable_relay = true

# Relay server address
relay_server = "relay.hauptbuch.org:8080"

# Enable encrypted connections
enable_encryption = true

# Encryption key file
encryption_key_file = "/var/lib/hauptbuch/network.key"

# Enable compression
enable_compression = true

# Compression level (1-9)
compression_level = 6
```

## Consensus Configuration

### Proof of Stake Settings

```toml
[consensus]
# Validator set size
validator_set_size = 100

# Block time in milliseconds
block_time_ms = 5000

# Epoch length in blocks
epoch_length_blocks = 1000

# VDF difficulty
vdf_difficulty = 1000000

# PoW difficulty (hex string)
pow_difficulty = "0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

# Enable slashing
slashing_enabled = true

# Slashing threshold (percentage)
slashing_threshold = 0.05

# Minimum stake amount
min_stake_amount = 1000000000000000000

# Maximum stake amount
max_stake_amount = 1000000000000000000000
```

### Advanced Consensus Settings

```toml
[consensus]
# Enable optimistic validation
optimistic_validation = true

# Enable block pre-confirmation
block_preconfirmation = true

# Enable transaction pre-confirmation
tx_preconfirmation = true

# Enable MEV protection
mev_protection = true

# MEV protection threshold
mev_protection_threshold = 0.1

# Enable cross-chain consensus
cross_chain_consensus = true

# Cross-chain consensus timeout
cross_chain_timeout = 30
```

## Cryptography Configuration

### Quantum-Resistant Settings

```toml
[crypto]
# Default signature scheme
default_signature_scheme = "ml-dsa"

# Default key exchange scheme
default_key_exchange_scheme = "ml-kem"

# Enable hybrid mode
hybrid_mode_enabled = true

# Classical signature fallback
classical_signature_fallback = "p256"

# Classical key exchange fallback
classical_key_exchange_fallback = "x25519"

# Enable quantum-resistant key generation
quantum_resistant_keygen = true

# Enable quantum-resistant encryption
quantum_resistant_encryption = true
```

### Advanced Cryptography Settings

```toml
[crypto]
# Enable zero-knowledge proofs
zk_proofs_enabled = true

# ZK proof system
zk_proof_system = "plonky3"

# Enable confidential transactions
confidential_tx_enabled = true

# Enable homomorphic encryption
homomorphic_encryption = true

# Enable secure multi-party computation
smpc_enabled = true

# Enable threshold cryptography
threshold_crypto = true

# Threshold size
threshold_size = 3
```

## Database Configuration

### RocksDB Settings

```toml
[database]
# Database type
type = "rocksdb"

# Database path
path = "/var/lib/hauptbuch/db"

# Cache size in MB
cache_size_mb = 256

# Maximum open files
max_open_files = 1000

# Enable compression
compression_enabled = true

# Compression algorithm
compression_algorithm = "lz4"

# Enable WAL
wal_enabled = true

# WAL size limit
wal_size_limit = 1000000000
```

### Advanced Database Settings

```toml
[database]
# Enable read-only mode
read_only = false

# Enable statistics
statistics_enabled = true

# Statistics output file
statistics_output = "/var/lib/hauptbuch/db_stats.json"

# Enable backup
backup_enabled = true

# Backup interval (seconds)
backup_interval = 3600

# Backup retention (days)
backup_retention = 7

# Enable replication
replication_enabled = false

# Replication servers
replication_servers = ["replica1.hauptbuch.org", "replica2.hauptbuch.org"]
```

## Monitoring Configuration

### Basic Monitoring

```toml
[monitoring]
# Enable monitoring
enabled = true

# Metrics address
metrics_address = "0.0.0.0:9090"

# Enable tracing
tracing_enabled = true

# Jaeger agent host
jaeger_agent_host = "127.0.0.1"

# Jaeger agent port
jaeger_agent_port = 6831

# Enable health checks
health_checks_enabled = true

# Health check interval (seconds)
health_check_interval = 30
```

### Advanced Monitoring

```toml
[monitoring]
# Enable performance profiling
profiling_enabled = true

# Profiling output directory
profiling_output_dir = "/var/lib/hauptbuch/profiles"

# Enable memory profiling
memory_profiling = true

# Memory profiling interval (seconds)
memory_profiling_interval = 60

# Enable CPU profiling
cpu_profiling = true

# CPU profiling interval (seconds)
cpu_profiling_interval = 60

# Enable network profiling
network_profiling = true

# Network profiling interval (seconds)
network_profiling_interval = 60
```

## Environment Variables

### Core Environment Variables

```bash
# Core Configuration
HAUPTBUCH_NETWORK_ID="hauptbuch-testnet-1"
HAUPTBUCH_CHAIN_ID=1337
HAUPTBUCH_LOG_LEVEL="info"
HAUPTBUCH_DATA_DIR="/var/lib/hauptbuch"
HAUPTBUCH_DEV_MODE=false
HAUPTBUCH_TESTNET=true
HAUPTBUCH_MAINNET=false

# Consensus Configuration
HAUPTBUCH_VALIDATOR_SET_SIZE=100
HAUPTBUCH_BLOCK_TIME_MS=5000
HAUPTBUCH_EPOCH_LENGTH_BLOCKS=1000
HAUPTBUCH_VDF_DIFFICULTY=1000000
HAUPTBUCH_POW_DIFFICULTY="0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
HAUPTBUCH_SLASHING_ENABLED=true
HAUPTBUCH_SLASHING_THRESHOLD=0.05
HAUPTBUCH_MIN_STAKE_AMOUNT=1000000000000000000
HAUPTBUCH_MAX_STAKE_AMOUNT=1000000000000000000000

# Network Configuration
HAUPTBUCH_LISTEN_ADDRESS="0.0.0.0:8080"
HAUPTBUCH_BOOTNODES="/ip4/127.0.0.1/tcp/8080/p2p/Qm..."
HAUPTBUCH_MAX_CONNECTIONS=100
HAUPTBUCH_ENABLE_QUIC=true
HAUPTBUCH_ENABLE_TCP=true
HAUPTBUCH_CONNECTION_TIMEOUT=30
HAUPTBUCH_KEEP_ALIVE_INTERVAL=60

# Cryptography Configuration
HAUPTBUCH_DEFAULT_SIGNATURE_SCHEME="ml-dsa"
HAUPTBUCH_DEFAULT_KEY_EXCHANGE_SCHEME="ml-kem"
HAUPTBUCH_HYBRID_MODE_ENABLED=true
HAUPTBUCH_CLASSICAL_SIGNATURE_FALLBACK="p256"
HAUPTBUCH_CLASSICAL_KEY_EXCHANGE_FALLBACK="x25519"
HAUPTBUCH_QUANTUM_RESISTANT_KEYGEN=true
HAUPTBUCH_QUANTUM_RESISTANT_ENCRYPTION=true

# Database Configuration
HAUPTBUCH_DATABASE_TYPE="rocksdb"
HAUPTBUCH_DATABASE_PATH="/var/lib/hauptbuch/db"
HAUPTBUCH_DATABASE_CACHE_SIZE_MB=256
HAUPTBUCH_DATABASE_MAX_OPEN_FILES=1000
HAUPTBUCH_DATABASE_COMPRESSION_ENABLED=true
HAUPTBUCH_DATABASE_COMPRESSION_ALGORITHM="lz4"
HAUPTBUCH_DATABASE_WAL_ENABLED=true
HAUPTBUCH_DATABASE_WAL_SIZE_LIMIT=1000000000

# Cache Configuration
HAUPTBUCH_CACHE_TYPE="redis"
HAUPTBUCH_CACHE_ADDRESS="127.0.0.1:6379"
HAUPTBUCH_CACHE_PASSWORD=""
HAUPTBUCH_CACHE_TTL_SECONDS=3600

# Monitoring Configuration
HAUPTBUCH_MONITORING_ENABLED=true
HAUPTBUCH_METRICS_ADDRESS="0.0.0.0:9090"
HAUPTBUCH_TRACING_ENABLED=true
HAUPTBUCH_JAEGER_AGENT_HOST="127.0.0.1"
HAUPTBUCH_JAEGER_AGENT_PORT=6831
HAUPTBUCH_HEALTH_CHECKS_ENABLED=true
HAUPTBUCH_HEALTH_CHECK_INTERVAL=30
```

## Advanced Configuration

### Layer 2 Configuration

```toml
[l2]
# Enable rollups
rollups_enabled = true

# Enable zkEVM
zkevm_enabled = true

# Enable SP1 zkVM
sp1_zkvm_enabled = true

# Enable Jolt zkVM
jolt_zkvm_enabled = false

# Enable EIP-4844
eip4844_enabled = true

# Enable KZG ceremony
kzg_ceremony_enabled = true
```

### Cross-Chain Configuration

```toml
[cross_chain]
# Enable bridge
bridge_enabled = true

# Enable IBC
ibc_enabled = true

# Enable CCIP
ccip_enabled = true

# Bridge address
bridge_address = "0x..."

# IBC client
ibc_client = "07-tendermint-0"

# CCIP router
ccip_router = "0x..."
```

### Account Abstraction Configuration

```toml
[account_abstraction]
# Enable ERC-4337
erc4337_enabled = true

# Enable ERC-6900
erc6900_enabled = true

# Enable ERC-7579
erc7579_enabled = true

# Enable ERC-7702
erc7702_enabled = true

# Paymaster address
paymaster_address = "0x..."

# Entry point address
entry_point_address = "0x..."
```

## Configuration Validation

### Validate Configuration

```bash
# Validate configuration file
hauptbuch config validate

# Validate environment variables
hauptbuch config validate-env

# Validate network configuration
hauptbuch config validate-network

# Validate consensus configuration
hauptbuch config validate-consensus

# Validate cryptography configuration
hauptbuch config validate-crypto
```

### Configuration Testing

```bash
# Test configuration
hauptbuch config test

# Test network connectivity
hauptbuch config test-network

# Test database connectivity
hauptbuch config test-database

# Test cache connectivity
hauptbuch config test-cache
```

## Best Practices

### Security

1. **Use strong passwords** for all authentication
2. **Enable encryption** for sensitive data
3. **Use secure key management** for cryptographic keys
4. **Enable monitoring** for security events
5. **Regular security audits** of configuration

### Performance

1. **Optimize database settings** for your workload
2. **Configure appropriate cache sizes** for your memory
3. **Use SSD storage** for better performance
4. **Enable compression** for network and storage
5. **Monitor performance metrics** regularly

### Reliability

1. **Enable backup and recovery** for critical data
2. **Configure appropriate timeouts** for operations
3. **Enable health checks** for monitoring
4. **Use redundant configurations** for critical services
5. **Test configurations** before production use

## Troubleshooting

### Common Issues

1. **Configuration validation errors**
   - Check syntax and format
   - Verify required fields
   - Check data types

2. **Network connectivity issues**
   - Verify network settings
   - Check firewall configuration
   - Test connectivity to bootnodes

3. **Database issues**
   - Check database permissions
   - Verify disk space
   - Check database configuration

4. **Performance issues**
   - Monitor resource usage
   - Optimize configuration settings
   - Check for bottlenecks

### Getting Help

- **Documentation**: Check the [documentation](../README.md)
- **Issues**: Report issues on [GitHub](https://github.com/hauptbuch/hauptbuch/issues)
- **Community**: Ask questions on [Discord](https://discord.gg/hauptbuch)
- **Support**: Contact support at [support@hauptbuch.org](mailto:support@hauptbuch.org)

## Conclusion

This configuration guide provides comprehensive instructions for setting up and optimizing Hauptbuch nodes. Follow the best practices and troubleshooting tips to ensure a reliable and secure deployment.
