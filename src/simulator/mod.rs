//! Cross-Chain Governance Simulator Module
//!
//! This module provides comprehensive simulation capabilities for cross-chain governance scenarios,
//! enabling researchers to model and study complex federated governance interactions across
//! multiple blockchain networks including Ethereum, Polkadot, and Cosmos.
//!
//! The simulator supports:
//! - Coordinated governance proposals across multiple chains
//! - Vote aggregation with variable participation rates and stake distributions
//! - Network condition simulation (delays, failures, forks)
//! - Real-time analytics and visualization
//! - Security validation with quantum-resistant cryptography

pub mod governance;

#[cfg(test)]
mod governance_test;
