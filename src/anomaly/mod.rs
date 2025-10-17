//! AI-Driven Anomaly Detection Module
//!
//! This module provides comprehensive anomaly detection capabilities for the decentralized
//! voting blockchain, identifying unusual patterns in voting, network activity, and cross-chain
//! interactions using lightweight statistical models and machine learning techniques.

pub mod detector;

#[cfg(test)]
pub mod anomaly_test;

// Re-export main types for easy access
pub use detector::{
    AnomalyConfig, AnomalyDetection, AnomalyDetector, AnomalyError, AnomalySeverity, AnomalyType,
    ClusterResult, ZScoreResult,
};
