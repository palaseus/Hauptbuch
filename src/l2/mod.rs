//! Layer 2 Rollup system for enhanced scalability
//!
//! This module implements both optimistic rollup and zkEVM Layer 2 solutions
//! that enable 10,000+ TPS through off-chain transaction batching, L1 commitments,
//! fraud proof challenges, and zero-knowledge proofs for instant finality.

pub mod eip4844;
pub mod jolt_zkvm;
pub mod kzg_ceremony;
pub mod rollup;
pub mod sp1_zkvm;
pub mod zkevm;

// Re-export optimistic rollup types
pub use rollup::{
    // Main functions
    create_rollup,
    execute_transaction,
    generate_fraud_proof,
    submit_batch,
    validate_state_transition,

    verify_fraud_proof,
    AccountState,
    FraudProof,
    FraudProofData,

    L1Commitment,
    // Error types
    L2Error,
    L2Result,
    // Transaction types
    L2Transaction,
    // Fraud proof system
    MerkleWitness,
    // Core rollup types
    OptimisticRollup,
    Sequencer,
    StateExecutor,

    StateRoot,

    // State management
    StateTrie,
    TransactionBatch,
    TransactionStatus,

    TransactionType,
};

// Re-export zkEVM types
pub use zkevm::{
    AccessListItem,
    BlobTransaction,
    EVMContext,
    EVMOpcode,
    // EVM types
    EVMTransaction,
    KZGData,

    LogEntry,
    ZkEVMAccountState,
    ZkEVMCircuitConfig,
    // Core zkEVM types
    ZkEVMEngine,
    // Error types
    ZkEVMError,
    ZkEVMExecutionResult,
    ZkEVMMetrics,

    ZkEVMProof,
    ZkEVMResult,
    ZkEVMType,
};

// Re-export EIP-4844 types
pub use eip4844::{
    // Utility functions
    generate_blob_versioned_hash,

    BlobData,
    BlobGasParams,
    BlobPoolMetrics,
    BlobSidecar,
    // Core EIP-4844 types
    BlobTransaction as EIP4844BlobTransaction,
    BlobTransactionPool,
    // Error types
    EIP4844Error,
    EIP4844Result,
    KZGCommitment,
    KZGMetrics,

    KZGProof,
    KZGSystem,
    KZGTrustedSetup,
};
pub use jolt_zkvm::{
    ExecutionLog,
    // Error types
    JoltError,
    JoltExecutionContext,
    JoltExecutionResult,
    JoltMetrics,

    JoltProgram,
    JoltProof,
    JoltResult,
    // Core Jolt zkVM types
    JoltVM,
    LogLevel,
    ProgramMetadata,
};

// Re-export SP1 zkVM types
pub use sp1_zkvm::{
    ExecutionStatus,

    ExecutionStep,
    // Program types
    MemoryLayout,
    MemoryPermissions,
    MemorySection,
    SP1Config,
    // Error types
    SP1Error,
    SP1ExecutionContext,
    SP1ExecutionResult,
    SP1Metrics,
    SP1Program,
    SP1Proof,
    SP1Result,
    // Core SP1 zkVM types
    SP1ZkVM,
    TrustedSetup,
};

// Re-export KZG ceremony types
pub use kzg_ceremony::{
    CeremonyMetrics,

    CeremonyParameters,
    // Ceremony types
    CeremonyState,
    ContributionStatus,

    KZGCeremonyConfig,
    // Core KZG ceremony types
    KZGCeremonyEngine,
    // Error types
    KZGCeremonyError,
    KZGCeremonyParticipant,
    KZGCeremonyResult,
    KZGCeremonyState,
    TrustedSetupParameters,
};
