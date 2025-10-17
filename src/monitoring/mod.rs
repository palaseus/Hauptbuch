//! Monitoring Module Declaration
//!
//! This module provides comprehensive monitoring capabilities for the decentralized
//! voting blockchain, tracking health and performance metrics across all system components.

pub mod monitor;

#[cfg(test)]
pub mod monitor_test;

// Re-export main types for easy access
pub use monitor::*;
