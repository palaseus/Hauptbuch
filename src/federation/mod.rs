//! Multi-Chain Federation Module
//!
//! This module implements a comprehensive multi-chain federation system that enables
//! the decentralized voting blockchain to participate in a federated network with
//! other blockchains (e.g., Ethereum, Polkadot, Cosmos) for cross-chain voting and
//! governance, enhancing its research value by demonstrating interoperability in a
//! federated ecosystem.
//!
//! Key features:
//! - Cross-chain vote aggregation from multiple chains
//! - Federated governance proposals across chains
//! - State synchronization using Merkle proofs
//! - Quantum-resistant cryptography for security
//! - Integration with existing modules (PoS, governance, sharding, etc.)

#[allow(clippy::module_inception)]
pub mod federation;

#[cfg(test)]
pub mod federation_test;

// Re-export main federation types for easy access
pub use federation::*;
