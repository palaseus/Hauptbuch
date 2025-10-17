# API Reference

## Overview

This document provides comprehensive API reference documentation for the Hauptbuch blockchain platform. It covers all public APIs, their parameters, return values, and usage examples.

## Table of Contents

- [Core API](#core-api)
- [Consensus API](#consensus-api)
- [Cryptography API](#cryptography-api)
- [Network API](#network-api)
- [Database API](#database-api)
- [Cross-Chain API](#cross-chain-api)
- [Governance API](#governance-api)
- [Account Abstraction API](#account-abstraction-api)
- [Layer 2 API](#layer-2-api)
- [Monitoring API](#monitoring-api)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Authentication](#authentication)

## Core API

### Client

```rust
use hauptbuch_sdk::{Client, ClientBuilder, SdkError};

pub struct Client {
    rpc_url: String,
    api_key: Option<String>,
    timeout: Duration,
}

impl Client {
    /// Create a new client
    pub fn new(rpc_url: &str) -> Result<Self, SdkError> {
        ClientBuilder::new()
            .rpc_url(rpc_url)
            .build()
    }

    /// Get network information
    pub async fn get_network_info(&self) -> Result<NetworkInfo, SdkError> {
        // Implementation
    }

    /// Get node status
    pub async fn get_node_status(&self) -> Result<NodeStatus, SdkError> {
        // Implementation
    }

    /// Get chain information
    pub async fn get_chain_info(&self) -> Result<ChainInfo, SdkError> {
        // Implementation
    }
}
```

### Account

```rust
use hauptbuch_sdk::{Account, AccountBuilder, QuantumResistantCrypto};

pub struct Account {
    address: String,
    private_key: Vec<u8>,
    public_key: Vec<u8>,
    crypto: QuantumResistantCrypto,
}

impl Account {
    /// Create a new account
    pub fn new() -> Result<Self, SdkError> {
        AccountBuilder::new()
            .generate_keypair()
            .build()
    }

    /// Create account from private key
    pub fn from_private_key(private_key: &[u8]) -> Result<Self, SdkError> {
        AccountBuilder::new()
            .private_key(private_key)
            .build()
    }

    /// Get account address
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Get account balance
    pub async fn get_balance(&self, client: &Client) -> Result<u64, SdkError> {
        client.get_balance(&self.address).await
    }

    /// Sign transaction
    pub fn sign_transaction(&self, transaction: &Transaction) -> Result<SignedTransaction, SdkError> {
        self.crypto.sign_transaction(transaction, &self.private_key)
    }
}
```

### Transaction

```rust
use hauptbuch_sdk::{Transaction, TransactionBuilder, SignedTransaction};

pub struct Transaction {
    from: String,
    to: String,
    value: u64,
    data: Vec<u8>,
    gas_limit: u64,
    gas_price: u64,
    nonce: u64,
}

impl Transaction {
    /// Create a new transaction
    pub fn new() -> TransactionBuilder {
        TransactionBuilder::new()
    }

    /// Get transaction hash
    pub fn hash(&self) -> String {
        // Implementation
    }

    /// Serialize transaction
    pub fn serialize(&self) -> Vec<u8> {
        // Implementation
    }

    /// Deserialize transaction
    pub fn deserialize(data: &[u8]) -> Result<Self, SdkError> {
        // Implementation
    }
}

pub struct SignedTransaction {
    transaction: Transaction,
    signature: Vec<u8>,
    public_key: Vec<u8>,
}

impl SignedTransaction {
    /// Verify transaction signature
    pub fn verify(&self) -> Result<bool, SdkError> {
        // Implementation
    }

    /// Get transaction hash
    pub fn hash(&self) -> String {
        self.transaction.hash()
    }
}
```

## Consensus API

### Consensus Engine

```rust
use hauptbuch_consensus::{ConsensusEngine, Block, Validator, ConsensusError};

pub struct ConsensusEngine {
    validator_set: Vec<Validator>,
    current_block: Option<Block>,
    mempool: Vec<Transaction>,
}

impl ConsensusEngine {
    /// Create a new consensus engine
    pub fn new() -> Self {
        Self {
            validator_set: Vec::new(),
            current_block: None,
            mempool: Vec::new(),
        }
    }

    /// Add validator to set
    pub fn add_validator(&mut self, validator: Validator) -> Result<(), ConsensusError> {
        // Implementation
    }

    /// Remove validator from set
    pub fn remove_validator(&mut self, address: &str) -> Result<(), ConsensusError> {
        // Implementation
    }

    /// Update validator set
    pub fn update_validator_set(&mut self, validators: Vec<Validator>) -> Result<(), ConsensusError> {
        // Implementation
    }

    /// Get validator set
    pub fn get_validator_set(&self) -> &Vec<Validator> {
        &self.validator_set
    }

    /// Create new block
    pub fn create_block(&mut self, transactions: Vec<Transaction>) -> Result<Block, ConsensusError> {
        // Implementation
    }

    /// Validate block
    pub fn validate_block(&self, block: &Block) -> Result<bool, ConsensusError> {
        // Implementation
    }

    /// Add transaction to mempool
    pub fn add_transaction(&mut self, transaction: Transaction) -> Result<(), ConsensusError> {
        // Implementation
    }

    /// Get mempool
    pub fn get_mempool(&self) -> &Vec<Transaction> {
        &self.mempool
    }
}
```

### Block

```rust
use hauptbuch_consensus::{Block, BlockHeader, BlockBody};

pub struct Block {
    header: BlockHeader,
    body: BlockBody,
}

pub struct BlockHeader {
    pub height: u64,
    pub timestamp: u64,
    pub previous_hash: String,
    pub merkle_root: String,
    pub validator_set_hash: String,
    pub consensus_data: Vec<u8>,
}

pub struct BlockBody {
    pub transactions: Vec<Transaction>,
    pub validator_signatures: Vec<ValidatorSignature>,
}

impl Block {
    /// Create a new block
    pub fn new(header: BlockHeader, body: BlockBody) -> Self {
        Self { header, body }
    }

    /// Get block header
    pub fn header(&self) -> &BlockHeader {
        &self.header
    }

    /// Get block body
    pub fn body(&self) -> &BlockBody {
        &self.body
    }

    /// Get block hash
    pub fn hash(&self) -> String {
        // Implementation
    }

    /// Validate block
    pub fn is_valid(&self) -> bool {
        // Implementation
    }

    /// Serialize block
    pub fn serialize(&self) -> Vec<u8> {
        // Implementation
    }

    /// Deserialize block
    pub fn deserialize(data: &[u8]) -> Result<Self, SdkError> {
        // Implementation
    }
}
```

## Cryptography API

### Quantum-Resistant Crypto

```rust
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa, HybridCrypto, CryptoError};

pub struct MLKem {
    private_key: Vec<u8>,
    public_key: Vec<u8>,
}

impl MLKem {
    /// Generate keypair
    pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Implementation
    }

    /// Encrypt message
    pub fn encrypt(message: &[u8], public_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Implementation
    }

    /// Decrypt message
    pub fn decrypt(ciphertext: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Implementation
    }
}

pub struct MLDsa {
    private_key: Vec<u8>,
    public_key: Vec<u8>,
}

impl MLDsa {
    /// Generate keypair
    pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Implementation
    }

    /// Sign message
    pub fn sign(message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Implementation
    }

    /// Verify signature
    pub fn verify(message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, CryptoError> {
        // Implementation
    }
}

pub struct SLHDsa {
    private_key: Vec<u8>,
    public_key: Vec<u8>,
}

impl SLHDsa {
    /// Generate keypair
    pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>), CryptoError> {
        // Implementation
    }

    /// Sign message
    pub fn sign(message: &[u8], private_key: &[u8]) -> Result<Vec<u8>, CryptoError> {
        // Implementation
    }

    /// Verify signature
    pub fn verify(message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, CryptoError> {
        // Implementation
    }
}

pub struct HybridCrypto {
    quantum_crypto: QuantumResistantCrypto,
    classical_crypto: ClassicalCrypto,
}

impl HybridCrypto {
    /// Create new hybrid crypto
    pub fn new() -> Self {
        Self {
            quantum_crypto: QuantumResistantCrypto::new(),
            classical_crypto: ClassicalCrypto::new(),
        }
    }

    /// Sign with hybrid crypto
    pub fn sign(&self, message: &[u8]) -> Result<HybridSignature, CryptoError> {
        // Implementation
    }

    /// Verify hybrid signature
    pub fn verify(&self, message: &[u8], signature: &HybridSignature) -> Result<bool, CryptoError> {
        // Implementation
    }
}
```

## Network API

### Network Manager

```rust
use hauptbuch_network::{NetworkManager, Peer, Message, NetworkError};

pub struct NetworkManager {
    peers: HashMap<String, Peer>,
    message_queue: Vec<Message>,
    network_config: NetworkConfig,
}

impl NetworkManager {
    /// Create new network manager
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
            message_queue: Vec::new(),
            network_config: NetworkConfig::default(),
        }
    }

    /// Connect to peer
    pub async fn connect_peer(&mut self, peer: Peer) -> Result<(), NetworkError> {
        // Implementation
    }

    /// Disconnect from peer
    pub async fn disconnect_peer(&mut self, address: &str) -> Result<(), NetworkError> {
        // Implementation
    }

    /// Send message to peer
    pub async fn send_message(&mut self, peer: &Peer, message: Message) -> Result<(), NetworkError> {
        // Implementation
    }

    /// Broadcast message to all peers
    pub async fn broadcast_message(&mut self, message: Message) -> Result<(), NetworkError> {
        // Implementation
    }

    /// Get connected peers
    pub fn get_peers(&self) -> &HashMap<String, Peer> {
        &self.peers
    }

    /// Get network status
    pub fn get_network_status(&self) -> NetworkStatus {
        // Implementation
    }
}
```

### Peer

```rust
use hauptbuch_network::{Peer, PeerStatus, PeerInfo};

pub struct Peer {
    address: String,
    public_key: Vec<u8>,
    status: PeerStatus,
    info: PeerInfo,
}

pub enum PeerStatus {
    Connected,
    Disconnected,
    Connecting,
    Error,
}

pub struct PeerInfo {
    pub node_id: String,
    pub version: String,
    pub capabilities: Vec<String>,
    pub last_seen: u64,
}

impl Peer {
    /// Create new peer
    pub fn new(address: &str) -> Self {
        Self {
            address: address.to_string(),
            public_key: Vec::new(),
            status: PeerStatus::Disconnected,
            info: PeerInfo::default(),
        }
    }

    /// Get peer address
    pub fn address(&self) -> &str {
        &self.address
    }

    /// Get peer status
    pub fn status(&self) -> &PeerStatus {
        &self.status
    }

    /// Get peer info
    pub fn info(&self) -> &PeerInfo {
        &self.info
    }

    /// Update peer info
    pub fn update_info(&mut self, info: PeerInfo) {
        self.info = info;
    }
}
```

## Database API

### Database Interface

```rust
use hauptbuch_database::{Database, DatabaseError, Transaction, Block};

pub trait Database {
    /// Get value by key
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, DatabaseError>;
    
    /// Put key-value pair
    fn put(&self, key: &str, value: &[u8]) -> Result<(), DatabaseError>;
    
    /// Delete key
    fn delete(&self, key: &str) -> Result<(), DatabaseError>;
    
    /// Check if key exists
    fn exists(&self, key: &str) -> Result<bool, DatabaseError>;
    
    /// Get all keys with prefix
    fn get_keys_with_prefix(&self, prefix: &str) -> Result<Vec<String>, DatabaseError>;
    
    /// Begin transaction
    fn begin_transaction(&self) -> Result<DatabaseTransaction, DatabaseError>;
    
    /// Compact database
    fn compact(&self) -> Result<(), DatabaseError>;
    
    /// Get database statistics
    fn get_stats(&self) -> Result<DatabaseStats, DatabaseError>;
}

pub struct DatabaseTransaction {
    operations: Vec<DatabaseOperation>,
}

pub enum DatabaseOperation {
    Put { key: String, value: Vec<u8> },
    Delete { key: String },
}

impl DatabaseTransaction {
    /// Add put operation
    pub fn put(&mut self, key: &str, value: &[u8]) {
        self.operations.push(DatabaseOperation::Put {
            key: key.to_string(),
            value: value.to_vec(),
        });
    }
    
    /// Add delete operation
    pub fn delete(&mut self, key: &str) {
        self.operations.push(DatabaseOperation::Delete {
            key: key.to_string(),
        });
    }
    
    /// Commit transaction
    pub fn commit(self) -> Result<(), DatabaseError> {
        // Implementation
    }
    
    /// Rollback transaction
    pub fn rollback(self) -> Result<(), DatabaseError> {
        // Implementation
    }
}
```

## Cross-Chain API

### Bridge

```rust
use hauptbuch_cross_chain::{Bridge, CrossChainTransaction, BridgeError};

pub struct Bridge {
    source_chain: String,
    target_chain: String,
    bridge_address: String,
}

impl Bridge {
    /// Create new bridge
    pub fn new(source_chain: &str, target_chain: &str) -> Self {
        Self {
            source_chain: source_chain.to_string(),
            target_chain: target_chain.to_string(),
            bridge_address: String::new(),
        }
    }

    /// Transfer asset
    pub async fn transfer_asset(&self, transaction: CrossChainTransaction) -> Result<String, BridgeError> {
        // Implementation
    }

    /// Get bridge status
    pub async fn get_bridge_status(&self) -> Result<BridgeStatus, BridgeError> {
        // Implementation
    }

    /// Get supported chains
    pub fn get_supported_chains(&self) -> Vec<String> {
        // Implementation
    }
}
```

### IBC

```rust
use hauptbuch_cross_chain::{IBC, IBCChannel, IBCPacket};

pub struct IBC {
    client_id: String,
    connection_id: String,
    channel_id: String,
}

impl IBC {
    /// Create new IBC connection
    pub fn new(client_id: &str, connection_id: &str, channel_id: &str) -> Self {
        Self {
            client_id: client_id.to_string(),
            connection_id: connection_id.to_string(),
            channel_id: channel_id.to_string(),
        }
    }

    /// Send packet
    pub async fn send_packet(&self, packet: IBCPacket) -> Result<String, IBCError> {
        // Implementation
    }

    /// Receive packet
    pub async fn receive_packet(&self, packet: IBCPacket) -> Result<String, IBCError> {
        // Implementation
    }

    /// Get channel status
    pub async fn get_channel_status(&self) -> Result<IBCChannelStatus, IBCError> {
        // Implementation
    }
}
```

## Governance API

### Governance Engine

```rust
use hauptbuch_governance::{GovernanceEngine, Proposal, Vote, GovernanceError};

pub struct GovernanceEngine {
    proposals: HashMap<u64, Proposal>,
    votes: HashMap<u64, Vec<Vote>>,
    governance_config: GovernanceConfig,
}

impl GovernanceEngine {
    /// Create new governance engine
    pub fn new() -> Self {
        Self {
            proposals: HashMap::new(),
            votes: HashMap::new(),
            governance_config: GovernanceConfig::default(),
        }
    }

    /// Submit proposal
    pub fn submit_proposal(&mut self, proposal: Proposal) -> Result<u64, GovernanceError> {
        // Implementation
    }

    /// Vote on proposal
    pub fn vote(&mut self, vote: Vote) -> Result<(), GovernanceError> {
        // Implementation
    }

    /// Execute proposal
    pub fn execute_proposal(&mut self, proposal_id: u64) -> Result<(), GovernanceError> {
        // Implementation
    }

    /// Get proposal
    pub fn get_proposal(&self, proposal_id: u64) -> Option<&Proposal> {
        self.proposals.get(&proposal_id)
    }

    /// Get proposal votes
    pub fn get_proposal_votes(&self, proposal_id: u64) -> Option<&Vec<Vote>> {
        self.votes.get(&proposal_id)
    }
}
```

## Account Abstraction API

### ERC-4337

```rust
use hauptbuch_account_abstraction::{ERC4337, UserOperation, Paymaster, EntryPoint};

pub struct ERC4337 {
    entry_point: EntryPoint,
    paymaster: Paymaster,
}

impl ERC4337 {
    /// Create new ERC-4337 instance
    pub fn new(entry_point: EntryPoint, paymaster: Paymaster) -> Self {
        Self { entry_point, paymaster }
    }

    /// Submit user operation
    pub async fn submit_user_operation(&self, op: UserOperation) -> Result<String, AccountAbstractionError> {
        // Implementation
    }

    /// Get user operation status
    pub async fn get_user_operation_status(&self, op_hash: &str) -> Result<UserOperationStatus, AccountAbstractionError> {
        // Implementation
    }

    /// Get account balance
    pub async fn get_account_balance(&self, account: &str) -> Result<u64, AccountAbstractionError> {
        // Implementation
    }
}
```

## Layer 2 API

### Rollups

```rust
use hauptbuch_l2::{Rollup, RollupConfig, RollupError};

pub struct Rollup {
    config: RollupConfig,
    sequencer: Sequencer,
    prover: Prover,
}

impl Rollup {
    /// Create new rollup
    pub fn new(config: RollupConfig) -> Self {
        Self {
            config,
            sequencer: Sequencer::new(),
            prover: Prover::new(),
        }
    }

    /// Submit transaction
    pub async fn submit_transaction(&self, tx: Transaction) -> Result<String, RollupError> {
        // Implementation
    }

    /// Generate proof
    pub async fn generate_proof(&self, batch: TransactionBatch) -> Result<Proof, RollupError> {
        // Implementation
    }

    /// Verify proof
    pub async fn verify_proof(&self, proof: &Proof) -> Result<bool, RollupError> {
        // Implementation
    }
}
```

## Monitoring API

### Metrics

```rust
use hauptbuch_monitoring::{Metrics, MetricType, MetricValue};

pub struct Metrics {
    metrics: HashMap<String, MetricValue>,
}

impl Metrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Record metric
    pub fn record_metric(&mut self, name: &str, value: MetricValue) {
        self.metrics.insert(name.to_string(), value);
    }

    /// Get metric value
    pub fn get_metric(&self, name: &str) -> Option<&MetricValue> {
        self.metrics.get(name)
    }

    /// Get all metrics
    pub fn get_all_metrics(&self) -> &HashMap<String, MetricValue> {
        &self.metrics
    }

    /// Export metrics
    pub fn export_metrics(&self) -> String {
        // Implementation
    }
}
```

## Error Handling

### Error Types

```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum SdkError {
    NetworkError(String),
    CryptoError(String),
    DatabaseError(String),
    ConsensusError(String),
    ValidationError(String),
    ConfigurationError(String),
    AuthenticationError(String),
    RateLimitError(String),
    TimeoutError(String),
    UnknownError(String),
}

impl Error for SdkError {}

impl fmt::Display for SdkError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SdkError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            SdkError::CryptoError(msg) => write!(f, "Crypto error: {}", msg),
            SdkError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            SdkError::ConsensusError(msg) => write!(f, "Consensus error: {}", msg),
            SdkError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            SdkError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            SdkError::AuthenticationError(msg) => write!(f, "Authentication error: {}", msg),
            SdkError::RateLimitError(msg) => write!(f, "Rate limit error: {}", msg),
            SdkError::TimeoutError(msg) => write!(f, "Timeout error: {}", msg),
            SdkError::UnknownError(msg) => write!(f, "Unknown error: {}", msg),
        }
    }
}
```

## Rate Limiting

### Rate Limiter

```rust
use hauptbuch_rate_limiter::{RateLimiter, RateLimitConfig};

pub struct RateLimiter {
    config: RateLimitConfig,
    requests: HashMap<String, Vec<Instant>>,
}

pub struct RateLimitConfig {
    pub max_requests: u64,
    pub time_window: Duration,
    pub burst_limit: u64,
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            requests: HashMap::new(),
        }
    }

    /// Check if request is allowed
    pub fn is_allowed(&mut self, key: &str) -> bool {
        // Implementation
    }

    /// Get remaining requests
    pub fn get_remaining_requests(&self, key: &str) -> u64 {
        // Implementation
    }

    /// Reset rate limiter
    pub fn reset(&mut self) {
        self.requests.clear();
    }
}
```

## Authentication

### API Key Authentication

```rust
use hauptbuch_auth::{ApiKeyAuth, AuthError};

pub struct ApiKeyAuth {
    api_key: String,
    permissions: Vec<String>,
}

impl ApiKeyAuth {
    /// Create new API key auth
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            permissions: Vec::new(),
        }
    }

    /// Authenticate request
    pub fn authenticate(&self, request: &Request) -> Result<(), AuthError> {
        // Implementation
    }

    /// Check permissions
    pub fn has_permission(&self, permission: &str) -> bool {
        self.permissions.contains(&permission.to_string())
    }
}
```

## Conclusion

This API reference provides comprehensive documentation for all Hauptbuch APIs. Use this reference to understand the available functionality and integrate with the Hauptbuch platform effectively.
