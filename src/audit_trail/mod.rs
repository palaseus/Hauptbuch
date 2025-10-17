//! Real-Time Audit Trail Module
//!
//! This module provides comprehensive audit trail capabilities for the decentralized
//! voting blockchain, maintaining a tamper-proof log of all activities including votes,
//! governance proposals, cross-chain messages, and system events.

pub mod audit;

// Re-export main types for convenience
pub use audit::{
    AuditEventType, AuditLogEntry, AuditMerkleTree, AuditQuery, AuditTrail, AuditTrailConfig,
    AuditTrailError, AuditTrailStatistics, MerkleNode,
};

#[cfg(test)]
pub mod audit_test;
