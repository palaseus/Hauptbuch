# Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying Hauptbuch nodes and networks in various environments, from development to production. Learn how to set up, configure, and maintain Hauptbuch deployments.

## Table of Contents

- [Deployment Environments](#deployment-environments)
- [Development Deployment](#development-deployment)
- [Staging Deployment](#staging-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Deployment](#cloud-deployment)
- [Container Deployment](#container-deployment)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Security Considerations](#security-considerations)
- [Troubleshooting](#troubleshooting)

## Deployment Environments

### Environment Types

1. **Development**: Local development and testing
2. **Staging**: Pre-production testing and validation
3. **Production**: Live network deployment
4. **Cloud**: Cloud-based deployment
5. **Container**: Containerized deployment

### Environment Requirements

| Environment | CPU | RAM | Storage | Network | Availability |
|-------------|-----|-----|---------|---------|--------------|
| Development | 2 cores | 4GB | 50GB | 10 Mbps | 99% |
| Staging | 4 cores | 8GB | 100GB | 100 Mbps | 99.5% |
| Production | 8+ cores | 16GB+ | 500GB+ | 1 Gbps+ | 99.9% |
| Cloud | Variable | Variable | Variable | Variable | 99.9% |
| Container | Variable | Variable | Variable | Variable | 99.9% |

## Development Deployment

### Local Development Setup

```bash
# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git
cd hauptbuch

# Install dependencies
./scripts/install-deps.sh

# Build the project
cargo build --release

# Start development environment
./scripts/dev.sh
```

### Development Configuration

```toml
# config.toml for development
[core]
network_id = "hauptbuch-dev"
chain_id = 1337
log_level = "debug"
data_dir = "./data"

[consensus]
validator_set_size = 10
block_time_ms = 2000
epoch_length_blocks = 100
vdf_difficulty = 100000
pow_difficulty = "0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
slashing_enabled = false
slashing_threshold = 0.05

[network]
listen_address = "127.0.0.1:8080"
bootnodes = []
max_connections = 50
enable_quic = true

[crypto]
default_signature_scheme = "ml-dsa"
default_key_exchange_scheme = "ml-kem"
hybrid_mode_enabled = true
classical_signature_fallback = "p256"
classical_key_exchange_fallback = "x25519"

[database]
type = "rocksdb"
path = "./data/db"
cache_size_mb = 64
max_open_files = 100

[cache]
type = "redis"
address = "127.0.0.1:6379"
password = ""
ttl_seconds = 3600

[monitoring]
enabled = true
metrics_address = "127.0.0.1:9090"
tracing_enabled = true
jaeger_agent_host = "127.0.0.1"
jaeger_agent_port = 6831
```

### Development Scripts

```bash
# Start development environment
./scripts/dev.sh

# Run tests
./scripts/test.sh

# Run benchmarks
./scripts/benchmark.sh

# Clean environment
./scripts/clean.sh

# Health check
./scripts/health-check.sh
```

## Staging Deployment

### Staging Environment Setup

```bash
# Create staging directory
mkdir -p /opt/hauptbuch/staging
cd /opt/hauptbuch/staging

# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git .

# Build for staging
cargo build --release --features staging

# Configure staging
cp config.toml.staging config.toml
```

### Staging Configuration

```toml
# config.toml for staging
[core]
network_id = "hauptbuch-staging"
chain_id = 1338
log_level = "info"
data_dir = "/opt/hauptbuch/staging/data"

[consensus]
validator_set_size = 50
block_time_ms = 3000
epoch_length_blocks = 500
vdf_difficulty = 500000
pow_difficulty = "0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
slashing_enabled = true
slashing_threshold = 0.05

[network]
listen_address = "0.0.0.0:8080"
bootnodes = [
    "/ip4/127.0.0.1/tcp/8080/p2p/Qm...",
    "/ip4/192.168.1.100/tcp/8080/p2p/Qm...",
]
max_connections = 100
enable_quic = true

[crypto]
default_signature_scheme = "ml-dsa"
default_key_exchange_scheme = "ml-kem"
hybrid_mode_enabled = true
classical_signature_fallback = "p256"
classical_key_exchange_fallback = "x25519"

[database]
type = "rocksdb"
path = "/opt/hauptbuch/staging/data/db"
cache_size_mb = 256
max_open_files = 1000

[cache]
type = "redis"
address = "127.0.0.1:6379"
password = "staging_password"
ttl_seconds = 3600

[monitoring]
enabled = true
metrics_address = "0.0.0.0:9090"
tracing_enabled = true
jaeger_agent_host = "127.0.0.1"
jaeger_agent_port = 6831
```

### Staging Deployment Script

```bash
#!/bin/bash
# deploy-staging.sh

echo "Deploying Hauptbuch to staging environment..."

# Stop existing services
sudo systemctl stop hauptbuch

# Backup current deployment
sudo cp -r /opt/hauptbuch/staging /opt/hauptbuch/staging.backup.$(date +%Y%m%d_%H%M%S)

# Update code
cd /opt/hauptbuch/staging
git pull origin staging

# Build new version
cargo build --release --features staging

# Update configuration
cp config.toml.staging config.toml

# Start services
sudo systemctl start hauptbuch

# Verify deployment
./scripts/health-check.sh

echo "Staging deployment complete."
```

## Production Deployment

### Production Environment Setup

```bash
# Create production directory
sudo mkdir -p /opt/hauptbuch/production
sudo chown hauptbuch:hauptbuch /opt/hauptbuch/production
cd /opt/hauptbuch/production

# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git .

# Build for production
cargo build --release --features production

# Configure production
cp config.toml.production config.toml
```

### Production Configuration

```toml
# config.toml for production
[core]
network_id = "hauptbuch-mainnet"
chain_id = 1
log_level = "info"
data_dir = "/opt/hauptbuch/production/data"

[consensus]
validator_set_size = 100
block_time_ms = 5000
epoch_length_blocks = 1000
vdf_difficulty = 1000000
pow_difficulty = "0x0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
slashing_enabled = true
slashing_threshold = 0.05

[network]
listen_address = "0.0.0.0:8080"
bootnodes = [
    "/ip4/127.0.0.1/tcp/8080/p2p/Qm...",
    "/ip4/192.168.1.100/tcp/8080/p2p/Qm...",
    "/ip4/192.168.1.101/tcp/8080/p2p/Qm...",
]
max_connections = 200
enable_quic = true

[crypto]
default_signature_scheme = "ml-dsa"
default_key_exchange_scheme = "ml-kem"
hybrid_mode_enabled = true
classical_signature_fallback = "p256"
classical_key_exchange_fallback = "x25519"

[database]
type = "rocksdb"
path = "/opt/hauptbuch/production/data/db"
cache_size_mb = 1024
max_open_files = 2000

[cache]
type = "redis"
address = "127.0.0.1:6379"
password = "production_password"
ttl_seconds = 3600

[monitoring]
enabled = true
metrics_address = "0.0.0.0:9090"
tracing_enabled = true
jaeger_agent_host = "127.0.0.1"
jaeger_agent_port = 6831
```

### Production Deployment Script

```bash
#!/bin/bash
# deploy-production.sh

echo "Deploying Hauptbuch to production environment..."

# Stop existing services
sudo systemctl stop hauptbuch

# Backup current deployment
sudo cp -r /opt/hauptbuch/production /opt/hauptbuch/production.backup.$(date +%Y%m%d_%H%M%S)

# Update code
cd /opt/hauptbuch/production
git pull origin main

# Build new version
cargo build --release --features production

# Update configuration
cp config.toml.production config.toml

# Start services
sudo systemctl start hauptbuch

# Verify deployment
./scripts/health-check.sh

echo "Production deployment complete."
```

### Systemd Service

```ini
# /etc/systemd/system/hauptbuch.service
[Unit]
Description=Hauptbuch Node
After=network.target

[Service]
Type=simple
User=hauptbuch
Group=hauptbuch
WorkingDirectory=/opt/hauptbuch/production
ExecStart=/opt/hauptbuch/production/target/release/hauptbuch
Restart=always
RestartSec=10
Environment=RUST_LOG=info
Environment=HAUPTBUCH_NETWORK_ID=hauptbuch-mainnet
Environment=HAUPTBUCH_CHAIN_ID=1
Environment=HAUPTBUCH_DATA_DIR=/opt/hauptbuch/production/data

[Install]
WantedBy=multi-user.target
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# Launch EC2 instance
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.large \
    --key-name hauptbuch-key \
    --security-group-ids sg-12345678 \
    --subnet-id subnet-12345678 \
    --user-data file://user-data.sh
```

#### User Data Script

```bash
#!/bin/bash
# user-data.sh

# Update system
yum update -y

# Install dependencies
yum install -y git curl wget unzip

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker

# Clone repository
git clone https://github.com/hauptbuch/hauptbuch.git /opt/hauptbuch
cd /opt/hauptbuch

# Build project
cargo build --release

# Configure and start
cp config.toml.aws config.toml
./target/release/hauptbuch
```

#### ECS Deployment

```yaml
# ecs-task-definition.json
{
  "family": "hauptbuch",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::123456789012:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::123456789012:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "hauptbuch",
      "image": "hauptbuch/hauptbuch:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "RUST_LOG",
          "value": "info"
        },
        {
          "name": "HAUPTBUCH_NETWORK_ID",
          "value": "hauptbuch-mainnet"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hauptbuch",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Deployment

#### GKE Deployment

```yaml
# gke-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hauptbuch
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hauptbuch
  template:
    metadata:
      labels:
        app: hauptbuch
    spec:
      containers:
      - name: hauptbuch
        image: hauptbuch/hauptbuch:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: "info"
        - name: HAUPTBUCH_NETWORK_ID
          value: "hauptbuch-mainnet"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: hauptbuch-service
spec:
  selector:
    app: hauptbuch
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

### Azure Deployment

#### AKS Deployment

```yaml
# aks-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hauptbuch
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hauptbuch
  template:
    metadata:
      labels:
        app: hauptbuch
    spec:
      containers:
      - name: hauptbuch
        image: hauptbuch/hauptbuch:latest
        ports:
        - containerPort: 8080
        env:
        - name: RUST_LOG
          value: "info"
        - name: HAUPTBUCH_NETWORK_ID
          value: "hauptbuch-mainnet"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: hauptbuch-service
spec:
  selector:
    app: hauptbuch
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

## Container Deployment

### Docker Deployment

#### Dockerfile

```dockerfile
# Dockerfile
FROM rust:1.70-slim-bullseye AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libclang-dev \
    cmake \
    pkg-config \
    libssl-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY . .

# Build the application
RUN cargo build --release

# Create final image
FROM debian:bullseye-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder stage
COPY --from=builder /app/target/release/hauptbuch ./hauptbuch

# Copy configuration
COPY config.toml ./config.toml

# Expose ports
EXPOSE 8080 9090

# Set entrypoint
CMD ["./hauptbuch"]
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  hauptbuch-node:
    build: .
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

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.37.0
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:9.0.0
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  hauptbuch_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Kubernetes Deployment

#### Kubernetes Manifest

```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hauptbuch
  labels:
    app: hauptbuch
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hauptbuch
  template:
    metadata:
      labels:
        app: hauptbuch
    spec:
      containers:
      - name: hauptbuch
        image: hauptbuch/hauptbuch:latest
        ports:
        - containerPort: 8080
          name: p2p
        - containerPort: 9090
          name: metrics
        env:
        - name: RUST_LOG
          value: "info"
        - name: HAUPTBUCH_NETWORK_ID
          value: "hauptbuch-mainnet"
        - name: HAUPTBUCH_CHAIN_ID
          value: "1"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: hauptbuch-data
          mountPath: /data
      volumes:
      - name: hauptbuch-data
        persistentVolumeClaim:
          claimName: hauptbuch-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: hauptbuch-service
spec:
  selector:
    app: hauptbuch
  ports:
  - name: p2p
    port: 8080
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hauptbuch-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
```

## Monitoring and Maintenance

### Monitoring Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'hauptbuch'
    static_configs:
      - targets: ['hauptbuch-node:9090']
```

### Health Checks

```bash
#!/bin/bash
# health-check.sh

echo "Performing Hauptbuch health checks..."

# Check if node is running
if curl -s http://localhost:8080/status > /dev/null; then
    echo "✓ Node is running"
else
    echo "✗ Node is not running"
    exit 1
fi

# Check network connectivity
if curl -s http://localhost:8080/network/status > /dev/null; then
    echo "✓ Network is connected"
else
    echo "✗ Network is not connected"
    exit 1
fi

# Check database
if curl -s http://localhost:8080/database/status > /dev/null; then
    echo "✓ Database is accessible"
else
    echo "✗ Database is not accessible"
    exit 1
fi

echo "All health checks passed"
```

### Backup and Recovery

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="./backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/hauptbuch_backup_$TIMESTAMP.tar.gz"

echo "Starting Hauptbuch backup..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Stop services
sudo systemctl stop hauptbuch

# Create backup
tar -czvf "$BACKUP_FILE" /opt/hauptbuch/production/data

# Start services
sudo systemctl start hauptbuch

echo "Backup complete: $BACKUP_FILE"
```

### Log Management

```bash
#!/bin/bash
# log-rotation.sh

# Rotate logs
sudo logrotate /etc/logrotate.d/hauptbuch

# Clean old logs
find /var/log/hauptbuch -name "*.log.*" -mtime +7 -delete

# Compress old logs
find /var/log/hauptbuch -name "*.log" -mtime +1 -exec gzip {} \;
```

## Security Considerations

### Network Security

1. **Firewall Configuration**
   ```bash
   # Allow only necessary ports
   sudo ufw allow 8080/tcp
   sudo ufw allow 9090/tcp
   sudo ufw deny 22/tcp
   ```

2. **SSL/TLS Configuration**
   ```toml
   [network]
   enable_ssl = true
   ssl_cert_file = "/etc/ssl/certs/hauptbuch.crt"
   ssl_key_file = "/etc/ssl/private/hauptbuch.key"
   ```

3. **Access Control**
   ```toml
   [security]
   allowed_ips = ["192.168.1.0/24", "10.0.0.0/8"]
   blocked_ips = ["192.168.1.100"]
   rate_limiting = true
   max_requests_per_minute = 100
   ```

### Data Security

1. **Encryption at Rest**
   ```toml
   [database]
   encryption_enabled = true
   encryption_key_file = "/etc/hauptbuch/db.key"
   ```

2. **Backup Encryption**
   ```bash
   # Encrypt backups
   gpg --symmetric --cipher-algo AES256 hauptbuch_backup.tar.gz
   ```

3. **Key Management**
   ```toml
   [crypto]
   key_management = "vault"
   vault_address = "https://vault.hauptbuch.org"
   vault_token = "s.1234567890abcdef"
   ```

## Troubleshooting

### Common Issues

1. **Node won't start**
   - Check configuration file syntax
   - Verify required directories exist
   - Check system resources

2. **Network connectivity issues**
   - Verify firewall settings
   - Check network configuration
   - Test connectivity to bootnodes

3. **Performance issues**
   - Monitor resource usage
   - Check database performance
   - Optimize configuration

4. **Security issues**
   - Review access logs
   - Check for unauthorized access
   - Update security configurations

### Debugging

```bash
# Enable debug logging
export RUST_LOG=debug

# Run with verbose output
./target/release/hauptbuch --verbose

# Check system resources
htop
iostat -x 1
netstat -tulpn
```

### Getting Help

- **Documentation**: Check the [documentation](../README.md)
- **Issues**: Report issues on [GitHub](https://github.com/hauptbuch/hauptbuch/issues)
- **Community**: Ask questions on [Discord](https://discord.gg/hauptbuch)
- **Support**: Contact support at [support@hauptbuch.org](mailto:support@hauptbuch.org)

## Conclusion

This deployment guide provides comprehensive instructions for deploying Hauptbuch in various environments. Follow the security considerations and troubleshooting tips to ensure a reliable and secure deployment.
