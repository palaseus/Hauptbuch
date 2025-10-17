//! VDF Module Declaration
//!
//! This module provides production-grade Verifiable Delay Function (VDF) implementations
//! for secure randomness generation in the decentralized voting blockchain.

pub mod engine;
pub mod wesolowski;

#[cfg(test)]
pub mod vdf_test;

// Re-export main types for easy access
pub use engine::*;
pub use wesolowski::*;
