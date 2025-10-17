# Working Dockerfile for Hauptbuch
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false hauptbuch

# Set working directory
WORKDIR /app

# Copy the pre-built binary
COPY target/release/hauptbuch /app/hauptbuch

# Copy configuration files
COPY config.toml ./
COPY env.example ./

# Create data directory
RUN mkdir -p /app/data && chown -R hauptbuch:hauptbuch /app

# Switch to non-root user
USER hauptbuch

# Expose ports
EXPOSE 30303 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f -X POST http://localhost:8080 -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"hauptbuch_getNetworkInfo","params":{},"id":1}' || exit 1

# Default command
CMD ["./hauptbuch"]
