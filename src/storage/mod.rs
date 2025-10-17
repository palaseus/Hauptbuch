//! Storage Module
//!
//! This module provides production-ready storage solutions for the blockchain system,
//! including persistent state storage with RocksDB and high-performance caching with Redis.
//!
//! Key features:
//! - RocksDB for persistent state storage
//! - Redis for high-performance caching and mempool
//! - Transaction batching and atomic operations
//! - State snapshots and recovery
//! - Performance monitoring and metrics

pub mod production_storage;

// Re-export production storage types
pub use production_storage::{
    CacheConnection,
    ConnectionStatus,

    // Connection types
    DatabaseConnection,
    // Core storage types
    ProductionStorageEngine,
    // Error types
    ProductionStorageError,
    ProductionStorageResult,
    StorageConfig,
    StorageEntry,
    // Entry types
    StorageEntryType,

    StorageMetadata,
    StorageMetrics,

    StorageSnapshot,
};
