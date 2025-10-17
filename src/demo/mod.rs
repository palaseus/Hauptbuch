//! Demonstration Module for Educational Use
//!
//! This module provides an interactive demonstration of the decentralized
//! voting blockchain's features for educational and research purposes.

pub mod demo_script;
#[cfg(test)]
pub mod demo_test;

// Re-export main types for convenience
pub use demo_script::{BlockchainDemo, DemoConfig, DemoError, DemoResults, ScenarioResult};

#[cfg(test)]
pub use demo_test::{DemoTestSuite, TestConfig, TestResult, TestResults};
