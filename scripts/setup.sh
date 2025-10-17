#!/bin/bash

# Hauptbuch Setup Script
# This script sets up the development environment for Hauptbuch

set -e

echo "🚀 Setting up Hauptbuch development environment..."

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust is not installed. Please install Rust first:"
    echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check Rust version
RUST_VERSION=$(rustc --version | cut -d' ' -f2)
echo "✅ Rust version: $RUST_VERSION"

# Install system dependencies
echo "📦 Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        pkg-config \
        libssl-dev \
        libclang-dev \
        cmake \
        git \
        curl \
        wget
elif [[ "$OSTYPE" == "darwin"* ]]; then
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew is not installed. Please install Homebrew first."
        exit 1
    fi
    brew install pkg-config openssl cmake git curl wget
fi

# Install Rust components
echo "🔧 Installing Rust components..."
rustup component add rustfmt clippy
rustup toolchain install nightly

# Install additional Rust tools
echo "🛠️ Installing Rust tools..."
cargo install cargo-watch cargo-expand cargo-audit cargo-deny

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p logs
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources
mkdir -p monitoring/rules

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file..."
    cp env.example .env
    echo "✅ Created .env file. Please update the values as needed."
fi

# Build the project
echo "🔨 Building Hauptbuch..."
cargo build

# Run tests
echo "🧪 Running tests..."
cargo test

# Run benchmarks
echo "📊 Running benchmarks..."
cargo bench

# Check for security vulnerabilities
echo "🔒 Checking for security vulnerabilities..."
cargo audit

# Format code
echo "🎨 Formatting code..."
cargo fmt

# Run clippy
echo "🔍 Running clippy..."
cargo clippy -- -D warnings

echo "✅ Hauptbuch setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your configuration"
echo "2. Run 'cargo run' to start the node"
echo "3. Run 'docker-compose up' to start the full stack"
echo "4. Visit http://localhost:3000 for Grafana monitoring"
echo ""
echo "Happy coding! 🎉"
