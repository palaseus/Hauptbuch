# Testing Guide

## Overview

This guide provides comprehensive instructions for testing Hauptbuch nodes, networks, and applications. Learn how to run tests, validate functionality, and ensure quality assurance.

## Table of Contents

- [Testing Types](#testing-types)
- [Unit Testing](#unit-testing)
- [Integration Testing](#integration-testing)
- [Performance Testing](#performance-testing)
- [Security Testing](#security-testing)
- [End-to-End Testing](#end-to-end-testing)
- [Test Automation](#test-automation)
- [Continuous Integration](#continuous-integration)
- [Test Data Management](#test-data-management)
- [Troubleshooting](#troubleshooting)

## Testing Types

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Performance Tests**: Test system performance and scalability
4. **Security Tests**: Test security vulnerabilities and compliance
5. **End-to-End Tests**: Test complete user workflows
6. **Regression Tests**: Test for regressions after changes
7. **Load Tests**: Test system under high load
8. **Stress Tests**: Test system limits and failure modes

### Test Environment Setup

```bash
# Create test environment
mkdir -p tests/{unit,integration,performance,security,e2e}
mkdir -p tests/data/{fixtures,mocks,generated}
mkdir -p tests/config/{test,staging,production}
```

## Unit Testing

### Basic Unit Tests

```rust
// tests/unit/consensus_test.rs
use hauptbuch_consensus::{ConsensusEngine, Block, Transaction, Validator};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_creation() {
        let mut consensus = ConsensusEngine::new();
        let block = consensus.create_block(vec![]);
        assert!(block.is_valid());
    }

    #[test]
    fn test_transaction_validation() {
        let mut consensus = ConsensusEngine::new();
        let tx = Transaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000);
        
        assert!(consensus.validate_transaction(&tx));
    }

    #[test]
    fn test_validator_set_update() {
        let mut consensus = ConsensusEngine::new();
        let validators = vec![
            Validator::new("0x1111", 1000),
            Validator::new("0x2222", 2000),
        ];
        
        consensus.update_validator_set(validators);
        assert_eq!(consensus.validator_count(), 2);
    }
}
```

### Cryptography Unit Tests

```rust
// tests/unit/crypto_test.rs
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa, HybridCrypto};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_kem_key_exchange() {
        let (alice_private, alice_public) = MLKem::generate_keypair();
        let (bob_private, bob_public) = MLKem::generate_keypair();
        
        let message = b"Hello, Hauptbuch!";
        let ciphertext = MLKem::encrypt(message, &bob_public);
        let decrypted = MLKem::decrypt(&ciphertext, &bob_private);
        
        assert_eq!(message, &decrypted[..]);
    }

    #[test]
    fn test_ml_dsa_signature() {
        let (private_key, public_key) = MLDsa::generate_keypair();
        let message = b"Hello, Hauptbuch!";
        
        let signature = MLDsa::sign(message, &private_key);
        assert!(MLDsa::verify(message, &signature, &public_key));
    }

    #[test]
    fn test_slh_dsa_signature() {
        let (private_key, public_key) = SLHDsa::generate_keypair();
        let message = b"Hello, Hauptbuch!";
        
        let signature = SLHDsa::sign(message, &private_key);
        assert!(SLHDsa::verify(message, &signature, &public_key));
    }

    #[test]
    fn test_hybrid_crypto() {
        let hybrid = HybridCrypto::new();
        let message = b"Hello, Hauptbuch!";
        
        let (quantum_private, quantum_public) = MLDsa::generate_keypair();
        let (classical_private, classical_public) = ECDSA::generate_keypair();
        
        let quantum_signature = MLDsa::sign(message, &quantum_private);
        let classical_signature = ECDSA::sign(message, &classical_private);
        
        assert!(MLDsa::verify(message, &quantum_signature, &quantum_public));
        assert!(ECDSA::verify(message, &classical_signature, &classical_public));
    }
}
```

### Network Unit Tests

```rust
// tests/unit/network_test.rs
use hauptbuch_network::{NetworkManager, Peer, Message, Protocol};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_connection() {
        let mut network = NetworkManager::new();
        let peer = Peer::new("127.0.0.1:8080");
        
        assert!(network.connect_peer(peer).is_ok());
    }

    #[test]
    fn test_message_serialization() {
        let message = Message::new()
            .set_type(MessageType::Block)
            .set_data(b"test data");
        
        let serialized = message.serialize();
        let deserialized = Message::deserialize(&serialized);
        
        assert_eq!(message, deserialized);
    }

    #[test]
    fn test_protocol_handshake() {
        let mut network = NetworkManager::new();
        let peer = Peer::new("127.0.0.1:8080");
        
        assert!(network.handshake(peer).is_ok());
    }
}
```

## Integration Testing

### Consensus Integration Tests

```rust
// tests/integration/consensus_integration_test.rs
use hauptbuch_consensus::{ConsensusEngine, Block, Transaction, Validator};
use hauptbuch_network::{NetworkManager, Peer};
use hauptbuch_crypto::{MLDsa, MLKem};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consensus_with_network() {
        let mut consensus = ConsensusEngine::new();
        let mut network = NetworkManager::new();
        
        // Add validators
        let validators = vec![
            Validator::new("0x1111", 1000),
            Validator::new("0x2222", 2000),
        ];
        consensus.update_validator_set(validators);
        
        // Create and validate block
        let block = consensus.create_block(vec![]);
        assert!(block.is_valid());
        
        // Broadcast block
        assert!(network.broadcast_block(&block).await.is_ok());
    }

    #[tokio::test]
    async fn test_transaction_flow() {
        let mut consensus = ConsensusEngine::new();
        let mut network = NetworkManager::new();
        
        // Create transaction
        let tx = Transaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000);
        
        // Validate transaction
        assert!(consensus.validate_transaction(&tx));
        
        // Add to mempool
        consensus.add_transaction(tx);
        
        // Create block with transaction
        let block = consensus.create_block(consensus.get_mempool());
        assert!(block.is_valid());
        
        // Broadcast block
        assert!(network.broadcast_block(&block).await.is_ok());
    }
}
```

### Cross-Chain Integration Tests

```rust
// tests/integration/cross_chain_integration_test.rs
use hauptbuch_cross_chain::{Bridge, IBC, CCIP, CrossChainTransaction};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_operations() {
        let mut bridge = Bridge::new("ethereum", "polygon");
        
        // Test asset transfer
        let tx = CrossChainTransaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000)
            .target_chain("polygon");
        
        assert!(bridge.transfer_asset(tx).await.is_ok());
    }

    #[tokio::test]
    async fn test_ibc_operations() {
        let mut ibc = IBC::new("cosmos", "osmosis");
        
        // Test IBC transfer
        let tx = CrossChainTransaction::new()
            .from("cosmos1...")
            .to("osmo1...")
            .value(1000)
            .target_chain("osmosis");
        
        assert!(ibc.transfer_asset(tx).await.is_ok());
    }

    #[tokio::test]
    async fn test_ccip_operations() {
        let mut ccip = CCIP::new("ethereum", "avalanche");
        
        // Test CCIP transfer
        let tx = CrossChainTransaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000)
            .target_chain("avalanche");
        
        assert!(ccip.transfer_asset(tx).await.is_ok());
    }
}
```

## Performance Testing

### Benchmark Tests

```rust
// tests/performance/benchmark_test.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hauptbuch_consensus::{ConsensusEngine, Block, Transaction};
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa};

fn benchmark_block_creation(c: &mut Criterion) {
    let mut consensus = ConsensusEngine::new();
    
    c.bench_function("block_creation", |b| {
        b.iter(|| {
            let block = consensus.create_block(vec![]);
            black_box(block)
        })
    });
}

fn benchmark_transaction_processing(c: &mut Criterion) {
    let mut consensus = ConsensusEngine::new();
    let transactions = (0..1000)
        .map(|i| Transaction::new()
            .from(format!("0x{:x}", i))
            .to(format!("0x{:x}", i + 1))
            .value(1000))
        .collect::<Vec<_>>();
    
    c.bench_function("transaction_processing", |b| {
        b.iter(|| {
            for tx in &transactions {
                consensus.validate_transaction(tx);
            }
        })
    });
}

fn benchmark_crypto_operations(c: &mut Criterion) {
    let (private_key, public_key) = MLDsa::generate_keypair();
    let message = b"Hello, Hauptbuch!";
    
    c.bench_function("ml_dsa_signature", |b| {
        b.iter(|| {
            let signature = MLDsa::sign(message, &private_key);
            black_box(signature)
        })
    });
    
    c.bench_function("ml_dsa_verification", |b| {
        let signature = MLDsa::sign(message, &private_key);
        b.iter(|| {
            let valid = MLDsa::verify(message, &signature, &public_key);
            black_box(valid)
        })
    });
}

criterion_group!(
    benches,
    benchmark_block_creation,
    benchmark_transaction_processing,
    benchmark_crypto_operations
);
criterion_main!(benches);
```

### Load Testing

```rust
// tests/performance/load_test.rs
use hauptbuch_consensus::{ConsensusEngine, Transaction};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[tokio::test]
async fn test_high_load_transactions() {
    let consensus = Arc::new(ConsensusEngine::new());
    let semaphore = Arc::new(Semaphore::new(1000));
    
    let handles = (0..10000)
        .map(|i| {
            let consensus = consensus.clone();
            let semaphore = semaphore.clone();
            
            tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                let tx = Transaction::new()
                    .from(format!("0x{:x}", i))
                    .to(format!("0x{:x}", i + 1))
                    .value(1000);
                
                consensus.validate_transaction(&tx);
            })
        })
        .collect::<Vec<_>>();
    
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_concurrent_block_creation() {
    let consensus = Arc::new(ConsensusEngine::new());
    let semaphore = Arc::new(Semaphore::new(100));
    
    let handles = (0..1000)
        .map(|_| {
            let consensus = consensus.clone();
            let semaphore = semaphore.clone();
            
            tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                
                let block = consensus.create_block(vec![]);
                assert!(block.is_valid());
            })
        })
        .collect::<Vec<_>>();
    
    for handle in handles {
        handle.await.unwrap();
    }
}
```

## Security Testing

### Vulnerability Testing

```rust
// tests/security/vulnerability_test.rs
use hauptbuch_consensus::{ConsensusEngine, Block, Transaction};
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_spending_prevention() {
        let mut consensus = ConsensusEngine::new();
        
        // Create transaction
        let tx = Transaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000);
        
        // Add transaction to mempool
        consensus.add_transaction(tx.clone());
        
        // Try to add same transaction again
        assert!(!consensus.add_transaction(tx));
    }

    #[test]
    fn test_invalid_signature_rejection() {
        let mut consensus = ConsensusEngine::new();
        
        // Create transaction with invalid signature
        let tx = Transaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000)
            .signature("invalid_signature");
        
        assert!(!consensus.validate_transaction(&tx));
    }

    #[test]
    fn test_insufficient_balance_rejection() {
        let mut consensus = ConsensusEngine::new();
        
        // Create transaction with insufficient balance
        let tx = Transaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000000); // More than available balance
        
        assert!(!consensus.validate_transaction(&tx));
    }

    #[test]
    fn test_malformed_block_rejection() {
        let mut consensus = ConsensusEngine::new();
        
        // Create malformed block
        let block = Block::new()
            .set_previous_hash("invalid_hash")
            .set_timestamp(0);
        
        assert!(!consensus.validate_block(&block));
    }
}
```

### Penetration Testing

```rust
// tests/security/penetration_test.rs
use hauptbuch_network::{NetworkManager, Peer, Message};
use hauptbuch_consensus::{ConsensusEngine, Block, Transaction};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sybil_attack_prevention() {
        let mut consensus = ConsensusEngine::new();
        let mut network = NetworkManager::new();
        
        // Try to create multiple fake validators
        let fake_validators = (0..1000)
            .map(|i| Validator::new(format!("0x{:x}", i), 1))
            .collect::<Vec<_>>();
        
        // Should reject fake validators
        assert!(!consensus.update_validator_set(fake_validators));
    }

    #[test]
    fn test_eclipse_attack_prevention() {
        let mut network = NetworkManager::new();
        
        // Try to connect to malicious peers
        let malicious_peers = (0..100)
            .map(|i| Peer::new(format!("127.0.0.{}:8080", i)))
            .collect::<Vec<_>>();
        
        for peer in malicious_peers {
            assert!(!network.connect_peer(peer));
        }
    }

    #[test]
    fn test_ddos_attack_prevention() {
        let mut network = NetworkManager::new();
        
        // Try to send many requests
        for i in 0..10000 {
            let message = Message::new()
                .set_type(MessageType::Ping)
                .set_data(format!("request_{}", i));
            
            // Should rate limit requests
            assert!(!network.send_message(message));
        }
    }
}
```

## End-to-End Testing

### Complete Workflow Tests

```rust
// tests/e2e/workflow_test.rs
use hauptbuch_consensus::{ConsensusEngine, Block, Transaction};
use hauptbuch_network::{NetworkManager, Peer};
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complete_transaction_flow() {
        // Initialize components
        let mut consensus = ConsensusEngine::new();
        let mut network = NetworkManager::new();
        
        // Create and validate transaction
        let tx = Transaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000);
        
        assert!(consensus.validate_transaction(&tx));
        
        // Add to mempool
        consensus.add_transaction(tx);
        
        // Create block
        let block = consensus.create_block(consensus.get_mempool());
        assert!(block.is_valid());
        
        // Broadcast block
        assert!(network.broadcast_block(&block).await.is_ok());
    }

    #[tokio::test]
    async fn test_cross_chain_transaction_flow() {
        // Initialize cross-chain components
        let mut bridge = Bridge::new("ethereum", "polygon");
        let mut consensus = ConsensusEngine::new();
        
        // Create cross-chain transaction
        let tx = CrossChainTransaction::new()
            .from("0x1234")
            .to("0x5678")
            .value(1000)
            .target_chain("polygon");
        
        // Validate transaction
        assert!(consensus.validate_transaction(&tx));
        
        // Execute cross-chain transfer
        assert!(bridge.transfer_asset(tx).await.is_ok());
    }

    #[tokio::test]
    async fn test_governance_proposal_flow() {
        // Initialize governance components
        let mut governance = GovernanceEngine::new();
        let mut consensus = ConsensusEngine::new();
        
        // Create proposal
        let proposal = Proposal::new()
            .set_title("Test Proposal")
            .set_description("Test Description")
            .set_author("0x1234");
        
        // Submit proposal
        assert!(governance.submit_proposal(proposal).is_ok());
        
        // Vote on proposal
        let vote = Vote::new()
            .set_voter("0x5678")
            .set_proposal_id(1)
            .set_choice(VoteChoice::Yes);
        
        assert!(governance.vote(vote).is_ok());
        
        // Execute proposal
        assert!(governance.execute_proposal(1).is_ok());
    }
}
```

## Test Automation

### Automated Test Suite

```bash
#!/bin/bash
# run-tests.sh

echo "Running Hauptbuch test suite..."

# Unit tests
echo "Running unit tests..."
cargo test --lib

# Integration tests
echo "Running integration tests..."
cargo test --test integration

# Performance tests
echo "Running performance tests..."
cargo bench

# Security tests
echo "Running security tests..."
cargo test --test security

# End-to-end tests
echo "Running end-to-end tests..."
cargo test --test e2e

echo "All tests completed."
```

### Test Configuration

```toml
# Cargo.toml
[dev-dependencies]
criterion = "0.5"
tokio = { version = "1.0", features = ["full"] }
proptest = "1.0"
quickcheck = "1.0"

[[bench]]
name = "consensus_bench"
harness = false

[[bench]]
name = "crypto_bench"
harness = false

[[bench]]
name = "network_bench"
harness = false
```

## Continuous Integration

### GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal
          override: true
      
      - name: Run unit tests
        run: cargo test --lib
      
      - name: Run integration tests
        run: cargo test --test integration
      
      - name: Run security tests
        run: cargo test --test security
      
      - name: Run end-to-end tests
        run: cargo test --test e2e
      
      - name: Run benchmarks
        run: cargo bench
      
      - name: Run clippy
        run: cargo clippy -- -D warnings
      
      - name: Run rustfmt
        run: cargo fmt --all --check
```

### Test Coverage

```yaml
# .github/workflows/coverage.yml
name: Coverage

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal
          override: true
      
      - name: Install cargo-tarpaulin
        run: cargo install cargo-tarpaulin
      
      - name: Run tests with coverage
        run: cargo tarpaulin --out Html
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./tarpaulin-report.html
```

## Test Data Management

### Test Fixtures

```rust
// tests/fixtures/test_data.rs
use hauptbuch_consensus::{Block, Transaction, Validator};
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa};

pub struct TestData {
    pub blocks: Vec<Block>,
    pub transactions: Vec<Transaction>,
    pub validators: Vec<Validator>,
    pub crypto_keys: CryptoKeys,
}

pub struct CryptoKeys {
    pub ml_kem_keys: Vec<(Vec<u8>, Vec<u8>)>,
    pub ml_dsa_keys: Vec<(Vec<u8>, Vec<u8>)>,
    pub slh_dsa_keys: Vec<(Vec<u8>, Vec<u8>)>,
}

impl TestData {
    pub fn new() -> Self {
        Self {
            blocks: Self::generate_blocks(),
            transactions: Self::generate_transactions(),
            validators: Self::generate_validators(),
            crypto_keys: Self::generate_crypto_keys(),
        }
    }

    fn generate_blocks() -> Vec<Block> {
        (0..100)
            .map(|i| Block::new()
                .set_height(i)
                .set_timestamp(i * 1000)
                .set_previous_hash(format!("hash_{}", i)))
            .collect()
    }

    fn generate_transactions() -> Vec<Transaction> {
        (0..1000)
            .map(|i| Transaction::new()
                .from(format!("0x{:x}", i))
                .to(format!("0x{:x}", i + 1))
                .value(1000))
            .collect()
    }

    fn generate_validators() -> Vec<Validator> {
        (0..100)
            .map(|i| Validator::new(format!("0x{:x}", i), 1000))
            .collect()
    }

    fn generate_crypto_keys() -> CryptoKeys {
        let ml_kem_keys = (0..100)
            .map(|_| MLKem::generate_keypair())
            .collect();
        
        let ml_dsa_keys = (0..100)
            .map(|_| MLDsa::generate_keypair())
            .collect();
        
        let slh_dsa_keys = (0..100)
            .map(|_| SLHDsa::generate_keypair())
            .collect();
        
        CryptoKeys {
            ml_kem_keys,
            ml_dsa_keys,
            slh_dsa_keys,
        }
    }
}
```

### Test Mocks

```rust
// tests/mocks/mock_network.rs
use hauptbuch_network::{NetworkManager, Peer, Message};
use std::collections::HashMap;

pub struct MockNetworkManager {
    peers: HashMap<String, Peer>,
    messages: Vec<Message>,
}

impl MockNetworkManager {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            messages: Vec::new(),
        }
    }

    pub fn add_peer(&mut self, peer: Peer) {
        self.peers.insert(peer.address().to_string(), peer);
    }

    pub fn send_message(&mut self, message: Message) -> Result<(), NetworkError> {
        self.messages.push(message);
        Ok(())
    }

    pub fn get_messages(&self) -> &Vec<Message> {
        &self.messages
    }
}
```

## Troubleshooting

### Common Test Issues

1. **Test Timeouts**
   - Increase timeout values
   - Check for deadlocks
   - Verify async operations

2. **Test Failures**
   - Check test data
   - Verify assertions
   - Review test logic

3. **Performance Issues**
   - Optimize test data
   - Use parallel execution
   - Monitor resource usage

4. **Flaky Tests**
   - Add retry logic
   - Fix race conditions
   - Improve test isolation

### Debugging Tests

```bash
# Run tests with debug output
RUST_LOG=debug cargo test

# Run specific test
cargo test test_name

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test --jobs 4
```

### Getting Help

- **Documentation**: Check the [documentation](../README.md)
- **Issues**: Report issues on [GitHub](https://github.com/hauptbuch/hauptbuch/issues)
- **Community**: Ask questions on [Discord](https://discord.gg/hauptbuch)
- **Support**: Contact support at [support@hauptbuch.org](mailto:support@hauptbuch.org)

## Conclusion

This testing guide provides comprehensive instructions for testing Hauptbuch components and systems. Follow the best practices and troubleshooting tips to ensure quality assurance and reliable testing.
