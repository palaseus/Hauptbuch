/// Sharding module for blockchain scalability
///
/// This module provides a comprehensive sharding architecture that partitions
/// the blockchain into smaller, parallel shards to process transactions and
/// votes more efficiently. It includes:
/// - Multi-shard architecture with parallel transaction processing
/// - Stake-weighted validator assignment to shards
/// - Cross-shard communication for multi-shard transactions
/// - State consistency with Merkle proofs and synchronization
/// - Integration with PoS consensus, P2P networking, and smart contracts
///
/// The sharding layer is designed for high-throughput voting blockchain
/// applications with optimal performance and security.
pub mod shard;

/// Re-export the main sharding functionality
pub use shard::*;
