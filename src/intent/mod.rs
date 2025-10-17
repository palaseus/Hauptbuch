//! Intent-Based Cross-Chain Architecture Module
//!
//! This module provides comprehensive intent-based cross-chain functionality
//! that allows users to express desired outcomes rather than specific
//! transaction steps, with a competitive solver network for optimal fulfillment.
//!
//! Key features:
//! - Intent expression and validation
//! - Solver network with competitive fulfillment
//! - Cross-Chain Interoperability Protocol (CCIP) integration
//! - MEV protection through intent-based design
//! - Multi-chain state synchronization
//! - Quantum-resistant cryptography for cross-chain security
//! - Automated intent resolution and execution

pub mod ai_enhanced_engine;
pub mod intent_engine;

// Re-export main types for convenience
pub use ai_enhanced_engine::{
    AIEnhancedIntent,
    // AI-enhanced intent types
    AIEnhancedIntentEngine,
    AIEnhancedIntentEngineMetrics,

    // Error types
    AIEnhancedIntentError,
    AIEnhancedIntentResult,
    AIEnhancedSolver,
    AIOptimizationEngine,
    AIOptimizationMetrics,
    CostOptimization,
    CostOptimizationType,
    ExecutionResult,
    IntentExecutionRecord,
    IntentMarketplace,
    IntentMarketplaceMetrics,
    ModelType,
    OptimizationData,
    OptimizationModel,
    OptimizationStrategy,
    PredictedMetrics,
    ReputationMetrics,
    ReputationSystem,
    ReputationWeights,
    RoutingSuggestion,
    SolverLearningData,
    SolverPerformancePredictions,
    SolverReputation,
    SolverSpecialization,
};
pub use intent_engine::{
    // Cross-chain types
    CCIPMessage,
    CCIPMessageStatus,
    ChainState,

    CrossChainState,
    ExecutionTransaction,
    // Fulfillment types
    FulfillmentProof,
    Intent,
    // Core intent types
    IntentEngine,
    // Metrics types
    IntentEngineMetrics,

    // Error types
    IntentError,
    IntentParameters,

    IntentResult,
    IntentStatus,
    IntentType,
    // Solver types
    Solver,
    SolverMetrics,

    // Token types
    TokenAmount,

    TransactionStatus,
};
