//! Verifiable Random Function (VRF) Module
//!
//! This module implements a comprehensive Verifiable Random Function (VRF) system
//! to provide cryptographically secure randomness for governance processes in the
//! decentralized voting blockchain. The VRF ensures unbiased, verifiable randomness
//! for validator selection, proposal prioritization, and cross-chain coordination.
//!
//! Key features:
//! - ECDSA-based and Dilithium-based VRF implementations
//! - Verifiable random outputs with cryptographic proofs
//! - Fairness metrics and bias detection
//! - Integration with PoS consensus, governance, and federation modules
//! - Chart.js visualizations for randomness analysis

pub mod randomness;

#[cfg(test)]
pub mod randomness_test;

// Re-export main VRF types for convenience
pub use randomness::{
    BiasDetection, FairnessMetrics, RandomnessDistribution, VRFAlgorithm, VRFConfig, VRFEngine,
    VRFError, VRFOutput, VRFProof, VRFPurpose, VRFRandomness, VRFRandomnessResult,
};
