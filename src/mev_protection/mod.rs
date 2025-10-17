//! MEV Protection and AI-Based Detection Module
//!
//! This module provides comprehensive MEV protection mechanisms including
//! encrypted mempool, Proposer-Builder Separation (PBS), and AI-based MEV
//! detection and redistribution to protect users from front-running and
//! sandwich attacks.
//!
//! Key features:
//! - Encrypted mempool with time-locked decryption
//! - Proposer-Builder Separation (PBS) architecture
//! - AI-based MEV detection and classification
//! - MEV redistribution mechanisms
//! - Commit-reveal schemes for transaction privacy
//! - Quantum-resistant cryptography for MEV protection
//! - Real-time MEV monitoring and alerting
//! - Cross-chain MEV protection coordination

pub mod mev_engine;

// Re-export main types for convenience
pub use mev_engine::{
    BlockBid,
    Builder,
    BuilderMetrics,
    // Transaction types
    EncryptedTransaction,
    MEVAttackType,
    MEVDetectionCapabilities,

    // MEV detection types
    MEVDetectionResult,
    // Core engine types
    MEVProtectionEngine,
    // Error types
    MEVProtectionError,
    MEVProtectionLevel,

    MEVProtectionMetrics,

    MEVProtectionResult,
    MEVRecipient,

    MEVRedistributionPlan,
    MEVRedistributionType,
    MEVSeverity,
    // PBS types
    Proposer,
    ProposerMetrics,
};
