#!/bin/bash

# Hauptbuch Test Dependencies Installation Script
# This script installs all required dependencies for running the complete test suite

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

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    print_error "This script must be run from the Hauptbuch project root directory"
    exit 1
fi

print_status "🚀 Installing Hauptbuch test dependencies..."

# Install Python dependencies
print_status "Installing Python dependencies..."
pip3 install pytest pytest-asyncio aiohttp websockets psutil eth-account maturin

# Install Node.js dependencies for contract testing
print_status "Installing Node.js dependencies..."
if [ -d "tests/contracts" ]; then
    cd tests/contracts
    if [ -f "package.json" ]; then
        npm install
        print_success "Node.js dependencies installed"
    else
        print_warning "No package.json found in tests/contracts, skipping Node.js dependencies"
    fi
    cd ../..
else
    print_warning "tests/contracts directory not found, skipping Node.js dependencies"
fi

# Build Rust binary
print_status "Building Rust binary..."
cargo build --release --bin hauptbuch
if [ $? -eq 0 ]; then
    print_success "Rust binary built successfully"
else
    print_error "Failed to build Rust binary"
    exit 1
fi

# Build Python crypto extension
print_status "Building Python crypto extension..."
if command -v maturin &> /dev/null; then
    maturin develop --release
    if [ $? -eq 0 ]; then
        print_success "Python crypto extension built successfully"
    else
        print_warning "Failed to build Python crypto extension - tests will use mock implementations"
    fi
else
    print_warning "maturin not found - Python crypto extension will not be built"
fi

# Verify installations
print_status "Verifying installations..."

# Test Python imports
python3 -c "
import sys
try:
    import pytest
    import aiohttp
    import websockets
    import psutil
    print('✓ Python dependencies verified')
except ImportError as e:
    print(f'✗ Python dependency error: {e}')
    sys.exit(1)
"

# Test Rust binary
if [ -f "target/release/hauptbuch" ]; then
    print_success "✓ Rust binary exists"
else
    print_error "✗ Rust binary not found"
    exit 1
fi

# Test Python crypto extension (optional)
python3 -c "
try:
    import hauptbuch_crypto
    print('✓ Python crypto extension available')
except ImportError:
    print('⚠ Python crypto extension not available - will use mock implementations')
" 2>/dev/null || print_warning "Python crypto extension not available - will use mock implementations"

# Test Node.js setup (if available)
if [ -d "tests/contracts" ] && [ -f "tests/contracts/package.json" ]; then
    cd tests/contracts
    if npm list &> /dev/null; then
        print_success "✓ Node.js dependencies verified"
    else
        print_warning "⚠ Node.js dependencies may not be properly installed"
    fi
    cd ../..
fi

print_success "✅ All dependencies installed successfully!"
print_status "You can now run the test suite with:"
print_status "  ./tests/run_all_tests.sh"
print_status "  or individual test files in the tests/ directory"