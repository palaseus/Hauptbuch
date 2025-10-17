//! Performance Optimization Module
//!
//! This module provides comprehensive performance optimization capabilities
//! including parallel transaction execution (Block-STM), state caching,
//! and QUIC networking for high-throughput, low-latency blockchain operations.
//!
//! Key features:
//! - Block-STM parallel transaction execution with optimistic concurrency control
//! - Multi-level state caching with intelligent eviction policies
//! - QUIC networking for low-latency, high-throughput communication
//! - Memory-efficient data structures and algorithms
//! - Performance monitoring and optimization
//! - Adaptive resource management
//! - Cache warming and prefetching
//! - Connection multiplexing and stream management

pub mod block_stm;
pub mod optimistic_validation;
pub mod quic_networking;
pub mod sealevel_parallel;
pub mod state_cache;

// Re-export main types for convenience
pub use block_stm::{
    BlockSTMConfig,
    // Block-STM types
    BlockSTMEngine,
    // Error types
    BlockSTMError,
    BlockSTMMetrics,
    BlockSTMResult,
    DependencyType,
    StateEntry,

    TransactionContext,
    TransactionDependency,
    TransactionResult,
    TransactionStatus,
};
pub use optimistic_validation::{
    ConflictInfo,
    ConflictSeverity,

    ConflictType,
    DependencyGraph,
    OptimisticValidationConfig,
    // Core optimistic validation types
    OptimisticValidationEngine,
    // Error types
    OptimisticValidationError,
    OptimisticValidationMetrics,
    OptimisticValidationResult,
    RollbackData,
    SpeculativeContext,
    SpeculativePhase,
};
pub use sealevel_parallel::{
    ConflictResolutionStrategy,

    ConflictResult,
    ExecutionLane,
    ExecutionResult,
    LaneMetrics,
    LaneStatus,
    SealevelConfig,
    // Sealevel parallel execution types
    SealevelEngine,
    // Error types
    SealevelError,
    SealevelMetrics,
    SealevelResult,
    StateAccess,
    StateAccessList,
    StateAccessType,
    ValidationStatus,
    VersionedStateEntry,
};

pub use state_cache::{
    CacheConfig,
    CacheEntry,
    CacheLevel,

    CacheStatistics,
    EvictionPolicy,
    // State cache types
    StateCacheEngine,
    // Error types
    StateCacheError,
    StateCacheResult,
};

pub use quic_networking::{
    ConnectionState,
    QUICConnection,
    QUICNetworkingConfig,
    // QUIC networking types
    QUICNetworkingEngine,
    // Error types
    QUICNetworkingError,
    QUICNetworkingMetrics,
    QUICNetworkingResult,
    QUICPacket,
    QUICPacketHeader,
    QUICPacketType,

    QUICStream,
    StreamState,
    StreamType,
};
