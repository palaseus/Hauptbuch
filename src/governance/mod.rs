//! Governance module for the decentralized voting blockchain
//!
//! This module provides governance functionality including proposal creation,
//! voting, and execution for protocol upgrades and parameter changes.

pub mod proposal;

// Re-export main types for easy access
pub use proposal::{
    ExecutionStatus, GovernanceConfig, GovernanceError, GovernanceProposalSystem, Proposal,
    ProposalExecution, ProposalStatus, ProposalType, VoteChoice, VoteRecord, VoteTally,
};
