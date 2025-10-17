//! Enhanced Testing Suite Module
//!
//! This module provides comprehensive testing capabilities including chaos engineering,
//! fuzzing, formal verification, and integration testing to ensure system reliability
//! and robustness under extreme conditions.
//!
//! Key features:
//! - Chaos engineering with controlled failure injection
//! - Property-based testing with fuzzing
//! - Formal verification of critical properties
//! - Integration testing across all modules
//! - Performance benchmarking and stress testing
//! - Security testing and vulnerability assessment
//! - Network partition and Byzantine fault tolerance testing
//! - State consistency verification

pub mod chaos_engineering;
pub mod formal_verification;
pub mod integration_tests;

// Re-export main types for convenience
pub use chaos_engineering::{
    ByzantineValidatorConfig,
    ChaosEngineeringConfig,
    // Chaos engineering types
    ChaosEngineeringEngine,
    // Error types
    ChaosEngineeringError,
    ChaosEngineeringMetrics,
    ChaosEngineeringResult,
    ChaosExperiment,
    ChaosExperimentResult,
    ChaosExperimentStatus,
    ChaosExperimentType,
    CrashFailureConfig,
    NetworkPartitionConfig,
    SystemStability,
};

pub use formal_verification::{
    AtomicProposition,
    Counterexample,
    FormalProperty,
    // Formal verification types
    FormalVerificationEngine,
    // Error types
    FormalVerificationError,
    FormalVerificationMetrics,

    FormalVerificationResult,
    LogicalOperator,
    ModelCheckingResult,
    ModelCheckingStatistics,
    PropertyType,
    State,
    TemporalFormula,
    TemporalOperator,
    Transition,
    VerificationStatus,
};

pub use integration_tests::{
    ExpectedOutcome,
    IntegrationActionType,
    // Error types
    IntegrationTestError,
    IntegrationTestResult,
    IntegrationTestResult as IntegrationTestResultType,
    IntegrationTestScenario,
    IntegrationTestStep,
    // Integration testing types
    IntegrationTestingEngine,
    IntegrationTestingMetrics,

    PerformanceMetrics,
    PerformanceRequirements,
    ScalabilityRequirements,
    SecurityMetrics,
    SecurityRequirements,
    StepResult,
    SuccessCriterion,
    SuccessCriterionType,
    TestCategory,
    TestOutcome,
    TestStatus,
    ValidationCriterion,
    ValidationCriterionType,
    ValidationResult,
    ValidationStatus,
};
