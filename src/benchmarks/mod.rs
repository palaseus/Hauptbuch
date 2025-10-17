//! Benchmarking modules for performance measurement and analysis
//!
//! This module provides comprehensive benchmarking capabilities for the decentralized
//! voting blockchain, including cross-chain performance measurement, stress testing,
//! and performance optimization analysis.

pub mod cross_chain;

// Re-export main types for convenience
pub use cross_chain::{
    BenchmarkResult, CrossChainBenchmarkConfig, CrossChainBenchmarkError, CrossChainBenchmarkSuite,
    CrossChainMetrics,
};

#[cfg(test)]
pub mod cross_chain_test;
