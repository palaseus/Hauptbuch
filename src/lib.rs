//! Hauptbuch Python FFI Library
//! 
//! This library provides Python bindings for the Hauptbuch blockchain
//! quantum-resistant cryptographic operations.

pub mod crypto;
pub mod network;
pub mod consensus;
pub mod account_abstraction;
pub mod l2;
pub mod cross_chain;
pub mod based_rollup;
pub mod da_layer;
pub mod tee;
pub mod oracle;
pub mod governance;
pub mod mev_protection;
pub mod sharding;
pub mod identity;
pub mod restaking;
pub mod performance;
pub mod monitoring;
pub mod security;
pub mod analytics;
pub mod anomaly;
pub mod audit_trail;
pub mod compliance;
pub mod execution;
pub mod federation;
pub mod game_theory;
pub mod intent;
pub mod portal;
pub mod sdk;
pub mod simulator;
pub mod storage;
pub mod testing;
pub mod ui;
pub mod vdf;
pub mod visualization;
pub mod vrf;
pub mod zkml;
pub mod zktls;

// Re-export main types for Python FFI
pub use crypto::*;
pub use network::*;
pub use consensus::*;