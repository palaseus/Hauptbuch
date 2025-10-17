//! Based Rollup and Shared Sequencer Network Module
//!
//! This module provides comprehensive based rollup functionality with a
//! decentralized shared sequencer network that eliminates centralized
//! sequencing and provides censorship resistance.
//!
//! Key features:
//! - Decentralized sequencer network with stake-based selection
//! - Censorship-resistant transaction ordering
//! - Based rollup architecture with L1 settlement
//! - Shared sequencer for multiple rollups
//! - MEV protection through decentralized sequencing
//! - Quantum-resistant cryptography for sequencer security
//! - Economic incentives and slashing mechanisms
//! - Cross-rollup transaction coordination

pub mod espresso_sequencer;
pub mod hotstuff_bft;
pub mod preconfirmations;
pub mod sequencer_network;

// Re-export main types for convenience
pub use espresso_sequencer::{
    AtomicTransactionBatch,
    BatchStatus,

    // Transaction types
    CrossRollupStatus,
    CrossRollupTransaction,
    EspressoSequencer,
    EspressoSequencerConfig,
    // Core Espresso sequencer types
    EspressoSequencerEngine,
    // Error types
    EspressoSequencerError,
    EspressoSequencerMetrics,

    EspressoSequencerResult,
};
pub use hotstuff_bft::{
    // Core HotStuff BFT types
    HotStuffBFTEngine,
    // Error types
    HotStuffBFTError,
    HotStuffBFTMetrics,

    HotStuffBFTResult,
    HotStuffBlock,
    HotStuffSequencer,
    HotStuffVote,
    QuorumCertificate,
    SlashingType,
    ViewChangeMessage,
};
pub use preconfirmations::{
    CrossRollupData,
    MEVProtectionLevel,
    Preconfirmation,
    PreconfirmationBatch,
    // Core preconfirmation types
    PreconfirmationEngine,
    // Error types
    PreconfirmationError,
    PreconfirmationMetrics,

    PreconfirmationResult,
    PreconfirmationStatus,
    ReorgProtection,
    ReorgProtectionLevel,
};
pub use sequencer_network::{
    BasedRollup,
    // Error types
    BasedRollupError,
    BasedRollupResult,
    // Fee types
    FeeParams,

    // Metrics types
    NetworkMetrics,

    RollupConfig,

    // Transaction and block types
    RollupTransaction,
    RollupType,
    SequencedBlock,

    // Sequencer types
    Sequencer,
    SequencerMetrics,

    SequencerStatus,
    // Core network types
    SharedSequencerNetwork,
    // Slashing types
    SlashingEvent,
    SlashingParams,

    SlashingReason,
};
