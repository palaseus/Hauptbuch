/// P2P networking module for the blockchain system
///
/// This module provides peer-to-peer networking capabilities including:
/// - Gossip-based protocol for block and transaction propagation
/// - Secure communication with end-to-end encryption
/// - Node discovery and authentication
/// - Blockchain state synchronization
/// - Integration with PoS consensus module
///
/// The networking layer is designed for low latency and bandwidth efficiency
/// while maintaining security against Sybil attacks and malicious behavior.
pub mod p2p;

/// Re-export the main P2P networking functionality
pub use p2p::*;
