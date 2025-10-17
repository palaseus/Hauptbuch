//! Data Availability Layer Module
//!
//! This module provides a comprehensive data availability layer for the decentralized
//! voting blockchain, supporting multiple DA providers with automatic fallback mechanisms.
//!
//! Key features:
//! - Celestia-style data availability sampling
//! - EigenDA integration for Ethereum-native DA
//! - Avail integration for multi-chain DA
//! - Unified abstraction layer with fallback mechanisms
//! - Merkle proof verification for data integrity
//! - Integration with L2 rollups and federated chains
//! - Performance metrics and monitoring
//! - Quantum-resistant cryptography for security
//! - Comprehensive test coverage

pub mod abstraction;
pub mod avail;
pub mod celestia;
pub mod dynamic_selection;
pub mod eigenda;

#[cfg(test)]
pub mod celestia_test;

// Re-export Celestia types
pub use celestia::{
    CelestiaDALayer, DataAvailabilityConfig, DataAvailabilityError, DataAvailabilityMetrics,
    DataAvailabilityResult, DataBlock, DataType, MerkleProof, SamplingResult, VerificationResult,
};

// Re-export EigenDA types
pub use eigenda::{
    EigenDAAVSConfig, EigenDABatch, EigenDABlob, EigenDAEngine, EigenDAError, EigenDAMetrics,
    EigenDAOperator, EigenDAOperatorMetrics, EigenDAResult, RewardParams, SlashingCondition,
    SlashingType,
};

// Re-export Avail types
pub use avail::{
    AvailDataBlock, AvailEngine, AvailError, AvailMetrics, AvailNetworkConfig, AvailResult,
    AvailSamplingProof, AvailValidator, AvailValidatorMetrics, ErasureParams, SamplingParams,
};

// Re-export abstraction layer types
pub use abstraction::{
    DAAbstractionEngine, DAAbstractionError, DAAbstractionMetrics, DAAbstractionResult, DAProvider,
    DAProviderConfig, DAProviderHealth, DAProviderStatus, ProviderMetadata, UnifiedDataBlock,
};

// Re-export dynamic selection types
pub use dynamic_selection::{
    CostOracleEntry,
    // Provider and strategy types
    DAProviderType,
    DASelectionCriteria,
    DASelectionResult,
    DataUrgency,
    // Error types
    DynamicDAError,
    // Metrics types
    DynamicDAMetrics,

    DynamicDAResult,
    DynamicDASelectionConfig,
    // Core dynamic selection types
    DynamicDASelectionEngine,
    FallbackAction,

    FallbackCondition,
    FallbackConditionType,
    HybridDAStrategy,
    PerformanceMetrics,
};
