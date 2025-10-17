//! Zero-Knowledge Machine Learning (zkML) Module
//!
//! This module provides verifiable machine learning computations using zero-knowledge proofs
//! (zk-SNARKs) to ensure model integrity, prediction privacy, and trustless execution.
//! It extends the predictive oracle with zkML capabilities for governance predictions.

pub mod engine;
pub mod ezkl_integration;
pub mod halo2_integration;
pub mod halo2_production;

#[cfg(test)]
pub mod zkml_test;

// Re-export main types for easy access
pub use engine::{
    ZkCircuitConfig, ZkMLConfig, ZkMLError, ZkMLModel, ZkMLPrediction, ZkMLPredictionType,
    ZkMLSystem, ZkProofType, ZkSNARKProof,
};

// Re-export Halo2 integration types
pub use halo2_integration::{
    Halo2CircuitConfig, Halo2Error, Halo2Metrics, Halo2ModelType, Halo2Proof, Halo2Result,
    Halo2ZkML,
};

// Re-export production Halo2 types
pub use halo2_production::{
    CeremonyParams, Halo2ProductionConfig, Halo2ProductionEngine, Halo2ProductionError,
    Halo2ProductionMetrics, Halo2ProductionProof, Halo2ProductionProvingKey, Halo2ProductionResult,
    Halo2ProductionVerificationKey, MLInferenceCircuit, TrustedSetupParams,
};

// Re-export EZKL integration types
pub use ezkl_integration::{
    EZKLCircuitParams,

    EZKLInferenceMetrics,

    EZKLInferenceResult,
    EZKLModel,
    EZKLModelConfig,
    // Model types
    EZKLModelType,
    EZKLProof,
    // Core EZKL zkML types
    EZKLZkMLEngine,
    EZKLZkMLEngineConfig,
    // Error types
    EZKLZkMLError,
    // Metrics types
    EZKLZkMLMetrics,

    EZKLZkMLResult,
};
