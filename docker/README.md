# Docker Configuration

This directory contains Docker-related files for the Hauptbuch blockchain.

## Files

- `Dockerfile` - Main production Dockerfile (in root directory)
- `Dockerfile.original` - Original Dockerfile (backup)
- `Dockerfile.rocksdb` - RocksDB-specific Dockerfile variant

## Usage

### Build and Run

```bash
# Build the image
docker build -t hauptbuch .

# Run the container
docker run -d --name hauptbuch-node \
  -p 8080:8080 -p 8081:8081 -p 30303:30303 \
  hauptbuch
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs hauptbuch-node
```

## Ports

- **8080** - RPC API endpoint
- **8081** - WebSocket endpoint  
- **30303** - P2P network port

## Health Checks

The container includes health checks that verify the RPC endpoint is responding correctly.

## Security

- Runs as non-root user (`hauptbuch`)
- Minimal base image (Debian slim)
- No unnecessary packages
- Proper file permissions
