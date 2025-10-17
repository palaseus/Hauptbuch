//! Cross-Chain Interoperability Module Declaration
//!
//! This module provides comprehensive cross-chain interoperability capabilities
//! for the decentralized voting blockchain, enabling secure data transfer and
//! asset management across different blockchain networks.

pub mod bridge;
pub mod ccip;
pub mod ibc;

#[cfg(test)]
pub mod bridge_test;

// Re-export main types for easy access
pub use bridge::*;
pub use ccip::{
    // Error types
    CCIPError,
    CCIPMessage,
    CCIPMessageStatus,
    CCIPMetrics,

    CCIPOracle,
    CCIPProgrammableTransfer,
    CCIPResult,
    // Core CCIP types
    CCIPRouter,
    CCIPTokenTransfer,
    CCIPTransferStatus,
    OracleStatus,
};
pub use ibc::{
    ChannelOrdering,
    ChannelState,
    ClientType,
    ConnectionState,
    Height,
    IBCAcknowledgment,
    IBCChannel,
    IBCClientState,
    IBCConnection,
    IBCConsensusState,
    // Error types
    IBCError,
    // Core IBC types
    IBCLightClient,
    IBCMetrics,

    IBCPacket,
    IBCProof,
    IBCResult,
    ProofType,
};
