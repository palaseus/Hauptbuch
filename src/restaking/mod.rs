//! Restaking Module for Shared Economic Security
//!
//! This module implements an EigenLayer-inspired restaking system that enables
//! governance tokens to be restaked across federated chains or L2 rollups to
//! provide shared economic security for the decentralized voting blockchain.
//!
//! Key features:
//! - Liquid staking tokens (LSTs) for restaked governance tokens
//! - Cross-chain restaking across federated chains and L2 rollups
//! - Slashing conditions for misbehavior detection
//! - Yield optimization based on stake amount, chain risk, and network participation
//! - Integration with PoS consensus, governance, federation, and monitoring modules
//! - SHA-3 for stake integrity and Dilithium3/5 for restaking signatures
//! - Chart.js-compatible JSON outputs for research visualization

pub mod eigen;

#[cfg(test)]
pub mod eigen_test;
