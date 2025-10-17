//! Security module for the Decentralized Voting Blockchain
//!
//! This module provides comprehensive security auditing capabilities including
//! static analysis, runtime monitoring, vulnerability scanning, and audit reporting.

pub mod audit;
pub mod comprehensive_audit;
pub mod formal_verification;

#[cfg(test)]
pub mod audit_test;

// Re-export main types for convenience
pub use audit::{
    utils, AuditConfig, AuditError, AuditMetrics, AuditReport, SecurityAuditor,
    VulnerabilityFinding, VulnerabilitySeverity, VulnerabilityType,
};

// Re-export comprehensive audit types
pub use comprehensive_audit::{
    AuditDepth,

    // Risk and audit types
    RiskLevel,
    SecurityAuditConfig,
    // Core comprehensive audit types
    SecurityAuditEngine,
    // Error types
    SecurityAuditError,
    SecurityAuditMetrics,

    SecurityAuditReport,
    SecurityAuditResult,
    SecurityVulnerability,
    // Vulnerability types
    VulnerabilityStatus,
};

// Re-export formal verification types
pub use formal_verification::{
    FormalVerificationConfig,
    // Core formal verification types
    FormalVerificationEngine,
    // Error types
    FormalVerificationError,
    FormalVerificationMetrics,

    FormalVerificationResult,
    // Verification types
    VerificationProperty,
    VerificationResult,
    VerificationStatus,
};
