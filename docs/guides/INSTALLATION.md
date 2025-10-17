# Installation Guide

## Overview

This guide provides detailed instructions for installing Hauptbuch on various platforms and environments. Choose the installation method that best suits your needs.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Platform-Specific Instructions](#platform-specific-instructions)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB
- **Storage**: 50GB free space
- **Network**: 10 Mbps internet connection
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+ with WSL2

### Recommended Requirements

- **CPU**: 4+ cores, 3.0 GHz
- **RAM**: 8GB+
- **Storage**: 100GB+ free space (SSD preferred)
- **Network**: 100 Mbps+ internet connection
- **OS**: Linux (Ubuntu 22.04+), macOS (12+), Windows 11+ with WSL2

### Production Requirements

- **CPU**: 8+ cores, 3.5 GHz
- **RAM**: 16GB+
- **Storage**: 500GB+ free space (NVMe SSD)
- **Network**: 1 Gbps+ internet connection
- **OS**: Linux (Ubuntu 22.04+ LTS)

## Installation Methods

### Method 1: Docker Installation (Recommended)

The easiest way to install Hauptbuch is using Docker:

```bash
# Clone the repository
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch

# Run the setup script
./scripts/setup.sh

# Start the services
./scripts/dev.sh
```

### Method 2: Binary Installation

Download pre-built binaries:

```bash
# Download the latest release
wget https://github.com/hauptbuch/hauptbuch/releases/latest/download/hauptbuch-linux-x86_64.tar.gz

# Extract the archive
tar -xzf hauptbuch-linux-x86_64.tar.gz

# Move to system path
sudo mv hauptbuch /usr/local/bin/

# Verify installation
hauptbuch --version
```

### Method 3: Source Installation

Build from source:

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

# Install the binary
sudo cp target/release/hauptbuch /usr/local/bin/
```

### Method 4: Package Manager Installation

#### Ubuntu/Debian

```bash
# Add the Hauptbuch repository
curl -fsSL https://apt.hauptbuch.org/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hauptbuch-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/hauptbuch-archive-keyring.gpg] https://apt.hauptbuch.org/ stable main" | sudo tee /etc/apt/sources.list.d/hauptbuch.list

# Update package list
sudo apt-get update

# Install Hauptbuch
sudo apt-get install hauptbuch
```

#### macOS

```bash
# Install using Homebrew
brew tap hauptbuch/hauptbuch
brew install hauptbuch
```

#### Windows

```powershell
# Install using Chocolatey
choco install hauptbuch

# Or using Scoop
scoop bucket add hauptbuch https://github.com/hauptbuch/scoop-bucket.git
scoop install hauptbuch
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

#### Prerequisites

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    cmake \
    libclang-dev \
    git \
    curl \
    wget \
    unzip

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### Installation

```bash
# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch

# Build from source
cargo build --release

# Install systemd service
sudo cp scripts/hauptbuch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable hauptbuch
```

### macOS

#### Prerequisites

```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install rust cmake pkg-config openssl git

# Install Docker Desktop
brew install --cask docker
```

#### Installation

```bash
# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch

# Build from source
cargo build --release

# Install binary
sudo cp target/release/hauptbuch /usr/local/bin/
```

### Windows

#### Prerequisites

1. Install Windows Subsystem for Linux (WSL2)
2. Install Docker Desktop
3. Install Visual Studio Build Tools
4. Install Git

#### Installation

```powershell
# Open PowerShell as Administrator
# Install Rust
Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile "rustup-init.exe"
.\rustup-init.exe

# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch

# Build from source
cargo build --release

# Install binary
Copy-Item target\release\hauptbuch.exe C:\Windows\System32\
```

### Docker Installation

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  hauptbuch-node:
    image: hauptbuch/hauptbuch:latest
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - RUST_LOG=info
      - HAUPTBUCH_NETWORK_ID=hauptbuch-testnet-1
      - HAUPTBUCH_CHAIN_ID=1337
    volumes:
      - hauptbuch_data:/data
    restart: unless-stopped

volumes:
  hauptbuch_data:
```

#### Run with Docker

```bash
# Start the services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f hauptbuch-node
```

## Configuration

### Basic Configuration

Create a configuration file:

```bash
# Copy example configuration
cp config.toml.example config.toml

# Edit configuration
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

Set up environment variables:

```bash
# Copy example environment file
cp env.example .env

# Edit environment variables
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

## Verification

### Check Installation

```bash
# Check version
hauptbuch --version

# Check configuration
hauptbuch config validate

# Check dependencies
hauptbuch deps check
```

### Test Node Startup

```bash
# Start node in test mode
hauptbuch --testnet --log-level debug

# Check node status
curl http://localhost:8080/status

# Check network connectivity
curl http://localhost:8080/network/status
```

### Run Health Checks

```bash
# Run comprehensive health checks
./scripts/health-check.sh

# Check Docker services
docker-compose ps

# Check logs
docker-compose logs -f hauptbuch-node
```

## Troubleshooting

### Common Issues

#### 1. Build Failures

**Issue**: Cargo build fails with dependency errors

**Solution**:
```bash
# Update Rust
rustup update

# Clean build cache
cargo clean

# Rebuild
cargo build --release
```

#### 2. Port Already in Use

**Issue**: Port 8080 is already in use

**Solution**:
```bash
# Find process using port
sudo lsof -i :8080

# Kill the process
sudo kill -9 <PID>

# Or change port in config
# Edit config.toml and change listen_address
```

#### 3. Permission Denied

**Issue**: Permission denied when accessing data directory

**Solution**:
```bash
# Create data directory
sudo mkdir -p /var/lib/hauptbuch

# Set permissions
sudo chown -R $USER:$USER /var/lib/hauptbuch

# Or run with sudo
sudo hauptbuch
```

#### 4. Docker Issues

**Issue**: Docker containers won't start

**Solution**:
```bash
# Check Docker status
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check Docker Compose
docker-compose --version
```

#### 5. Network Connectivity

**Issue**: Can't connect to network

**Solution**:
```bash
# Check firewall
sudo ufw status

# Allow port 8080
sudo ufw allow 8080

# Check network configuration
ip addr show
```

### Getting Help

- **Documentation**: Check the [documentation](../README.md)
- **Issues**: Report issues on [GitHub](https://github.com/hauptbuch/hauptbuch/issues)
- **Community**: Ask questions on [Discord](https://discord.gg/hauptbuch)
- **Support**: Contact support at [support@hauptbuch.org](mailto:support@hauptbuch.org)

## Next Steps

After successful installation:

1. **Configure your node**: See [Configuration Guide](CONFIGURATION.md)
2. **Start developing**: See [Getting Started Guide](GETTING-STARTED.md)
3. **Deploy to production**: See [Deployment Guide](DEPLOYMENT.md)
4. **Join the community**: [Discord](https://discord.gg/hauptbuch)

## Conclusion

You've successfully installed Hauptbuch! The platform is now ready for development, testing, or production use. Continue with the configuration and getting started guides to begin using the platform.
