# Hauptbuch Blockchain Test Suite

This directory contains comprehensive tests for the Hauptbuch blockchain platform, covering all documented functionality through automated testing, integration validation, and performance benchmarking.

## Directory Structure

- `infrastructure/` - Network setup and deployment scripts
- `integration/` - Integration test scripts for blockchain functionality
- `contracts/` - Smart contract tests using Hardhat/Foundry
- `api/` - RPC/API interaction tests
- `performance/` - Performance benchmarks and stress tests
- `security/` - Security validation tests
- `fixtures/` - Test data and fixtures
- `utils/` - Shared testing utilities and Python SDK
- `ci/` - CI/CD configuration and automation

## Quick Start

```bash
# Run all tests
./run_all_tests.sh

# Run specific test categories
./run_all_tests.sh --category integration
./run_all_tests.sh --category performance
./run_all_tests.sh --category security

# Run with coverage
./run_all_tests.sh --coverage

# Run specific test file
python -m pytest tests/integration/test_consensus.py -v
```

## Test Categories

### Core Blockchain Tests
- Consensus mechanism validation
- Quantum-resistant cryptography
- Transaction processing
- Network operations

### Advanced Features Tests
- Account abstraction (ERC-4337/6900/7579/7702)
- Layer 2 scaling (rollups, zkEVM, SP1)
- Cross-chain interoperability
- Based rollup architecture
- Data availability layers

### Smart Contract Tests
- Voting contract with zk-SNARKs
- Governance token functionality
- Contract deployment and interaction

### Performance & Security Tests
- Throughput benchmarking
- Security vulnerability testing
- Stress testing under load
- MEV protection validation

## Requirements

- Python 3.8+
- Rust 1.70+
- Node.js 16+ (for contract testing)
- Docker (for local network deployment)
- 8GB+ RAM recommended

## Configuration

Tests use environment variables for configuration:

```bash
export HAUPTBUCH_RPC_URL="http://localhost:8080"
export HAUPTBUCH_NETWORK_ID="hauptbuch-testnet-1"
export HAUPTBUCH_CHAIN_ID=1337
export HAUPTBUCH_PRIVATE_KEY="your_private_key_here"
```

## Test Reports

Test results are generated in HTML format with detailed metrics, coverage analysis, and performance benchmarks. Reports are saved to `test-results/` directory.
